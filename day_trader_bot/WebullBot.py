import sys
sys.path.append('../cs229-master')
from preprocessing import main
import numpy as np
from webull import paper_webull
from torchvision.models import resnet50
import torch
import sched
import matplotlib.pyplot as plt
from data import draw_chart
import time
from PIL import Image
from torchvision import transforms
import tensorflow as tf
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
import datetime
from pytz import timezone
import os
tz = timezone('EST')
plt.switch_backend('agg')

tp = ThreadPoolExecutor(10)  # max 10 threads


def get_time(x):
    return (int(x.split(':')[0]) - 6) + int(x.split(':')[1]) / 60 - 0.5


def normalize(temp_x, temp_y):
    temp_x = temp_x.copy()
    temp_y = temp_y.copy()
    temp_x['Volume'] /= temp_x['Volume'].max()

    min_low = temp_x['Low'].min()
    for c in temp_x.columns:
        if ((c.startswith('SMA') or c.startswith('EMA')) and not c.endswith('Cross')) or c in ['Open', 'Close', 'High',
                                                                                               'Low']:
            temp_x[c] -= min_low

    max_high = temp_x['High'].max()
    for c in temp_x.columns:
        if ((c.startswith('SMA') or c.startswith('EMA')) and not c.endswith('Cross')) or c in ['Open', 'Close', 'High',
                                                                                               'Low']:
            temp_x[c] /= max_high

    temp_y -= min_low
    temp_y /= max_high
    return temp_x, temp_y


def threaded(fn):
    def wrapper(*args, **kwargs):
        return tp.submit(fn, *args, **kwargs)  # returns Future object

    return wrapper


def load_model(model_file):
    model_res = resnet50(pretrained=False, num_classes=3)
    model_res.load_state_dict(torch.load(model_file))
    # if torch.cuda.is_available():
    #     model_res = model_res.cuda()
    return model_res


class WebullBot:
    interval = 'm1'
    count = 1000
    delay = 60
    max_hold = 32
    URL = "https://api.stocktwits.com/api/2/trending/symbols.json?access_token=c8246a247edb65d655d4e1cd3781df676b4a8f76"
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    lookback = 60
    predict_size = 5

    def __init__(self, creds, model_file, symbols=None, use_mfa=False):
        if use_mfa:
            self.mfa_code = input('Enter MFA Code : ')
        else:
            self.mfa_code = None
        self.wb = paper_webull()
        # self.model_res = load_model(model_file)
        self.model_tf = tf.keras.models.load_model('../cs229-master/first_model/', compile=True)
        self.login_info = self.login_webull(creds)
        self.symbols = {}
        for order in self.wb.get_account()['positions']:
            self.symbols[order['ticker']['symbol']] = {"entered_trade": {'time': order['time'], 'price': order['ticker']['price']}, 'hist': pd.DataFrame()}

        if symbols is None:
            self.update_symbols()
            self.run_update_symbols = True
        else:
            self.run_update_symbols = False

        self.date = datetime.datetime.now(tz).isoformat().split('T')[0]
        self.s = sched.scheduler(time.time, time.sleep)
        self.s.enter(1, 1, self.run)

    def login_webull(self, creds):
        if self.mfa_code is not None:
            self.wb.get_mfa(creds['email'])  # mobile number should be okay as well.
            login_info = self.wb.login(creds['email'], creds['password'], 'My Device', self.mfa_code)
        else:
            login_info = self.wb.login(creds['email'], creds['password'])

        return login_info

    @threaded
    def get_bars(self, symbol):
        try:
            hist = self.wb.get_bars(stock=symbol, interval=self.interval, count=self.count, extendTrading=0)
            if hist.iloc[0]['close'] > 30:
                return None
            hist.columns = [c.capitalize() for c in hist.columns]
            hist.index.name = 'Local time'
            return pd.DataFrame(hist)
        except Exception as e:
            print(f'failed getting data for {symbol}')
            print(e)
            return None

    def predict(self, image_files):
        imgs = [self.TRANSFORM_IMG(Image.open(image_file).convert("RGB")).reshape((1, 3, 256, 256))
                for image_file in image_files]
        imgs = torch.cat(imgs, 0)
        outputs = self.model_res(imgs)
        probs = torch.nn.functional.softmax(outputs.data).tolist()
        _, preds = torch.max(outputs.data, 1)
        labels = preds.tolist()
        return labels, probs

    def place_order(self, symbol, pred, current, price):
        if not self.symbols[symbol]['entered_trade']:
            if pred[0] - current > 0.05:
                order = self.wb.place_order(stock=symbol, action='BUY', orderType='MKT', enforce='DAY', quant=1)
                print(f"BOUGHT: {symbol} --- Order#: {order.get('orderId')}")
                self.symbols[symbol]['entered_trade'] = {'time': datetime.datetime.now(), 'price': price}
                # decrease delay to check for sell opportunities more often
                self.delay = 5
        else:
            profit = (price - self.symbols[symbol]['entered_trade']['price']) / self.symbols[symbol]['entered_trade']['price']
            if pred[0] - current < 0.01:
                order = self.wb.place_order(stock=symbol, action='SELL', orderType='MKT', enforce='DAY', quant=1)
                print(f"SOLD: {symbol} --- Order#: {order.get('orderId')}")
                # if this is the only trade left then increase delay
                self.symbols[symbol]['entered_trade'] = False
            elif datetime.datetime.now() - self.symbols[symbol]['entered_trade']['time'] > datetime.timedelta(minutes=self.max_hold):
                order = self.wb.place_order(stock=symbol, action='SELL', orderType='MKT', enforce='DAY', quant=1)
                print(f"SOLD: {symbol} --- Order#: {order.get('orderId')} --- {self.max_hold} MINUTES EXPIRED")
                self.symbols[symbol]['entered_trade'] = False
            elif profit < -0.02:
                order = self.wb.place_order(stock=symbol, action='SELL', orderType='MKT', enforce='DAY', quant=1)
                print(f"SOLD: {symbol} --- Order#: {order.get('orderId')} --- LOST {profit}%")
                self.symbols[symbol]['entered_trade'] = False

    def update_symbols(self):
        r = requests.get(self.URL).json()
        new_symbols = [rr['symbol'] for rr in r['symbols'] if '.X' not in rr['symbol']]
        del_symbols = []
        self.delay = 60
        for symbol, values in self.symbols.items():
            if symbol not in new_symbols and not values['entered_trade']:
                del_symbols.append(symbol)
            elif symbol in new_symbols and values['entered_trade']:
                self.delay = 5

        for symbol in del_symbols:
            self.save_data(symbol)
            del self.symbols[symbol]

        for symbol in new_symbols:
            if symbol not in self.symbols:
                self.symbols[symbol] = {"entered_trade": False, 'hist': pd.DataFrame()}

    def save_data(self, symbol):
        values = self.symbols[symbol]
        filename = f"{symbol}_{self.date}"
        if os.path.exists(f"./live_data/{filename}.csv"):
            old_hist = pd.read_csv(f"./live_data/{filename}.csv").set_index('Local time')
            old_hist = pd.concat([old_hist, values['hist'][[idx not in old_hist.index for idx in values['hist'].index]]])
            if old_hist.shape[0] > 0:
                old_hist.to_csv(f"./live_data/{filename}.csv")
        else:
            if values['hist'].shape[0] > 0:
                values['hist'].to_csv(f"./live_data/{filename}.csv", index=True)
        return f"./live_data/{filename}.csv"

    def run(self):
        hists = {symbol: self.get_bars(symbol) for symbol in self.symbols.keys()}
        hists = {symbol: temp.result() for symbol, temp in hists.items()}
        filenames = []
        for symbol, hist in hists.items():
            if hist is None:
                filenames.append(self.save_data(symbol))
                del self.symbols[symbol]
                continue
            old_hist = self.symbols[symbol]['hist']
            self.symbols[symbol]['hist'] = pd.concat([old_hist, hist[[idx not in old_hist for idx in hist.index]]])

        # image_files = [draw_chart(symbol, values['hist']) for symbol, values in self.symbols.items()]

        # labels, probs = self.predict(image_files)
        data = []
        prices = []
        for symbol, hist in hists.items():
            if hist is not None:
                hist['Local time'] = [idx.isoformat().replace('T', ' ') for idx in hist.index]
                df = main(hist)
                for c in ['Unnamed: 0', 'Date', 'Class']:
                    if c in df.columns:
                        del df[c]

                df['Minute'] = df['Minute'].apply(get_time)

                actual_x, _ = df.iloc[-self.lookback:].values, df.iloc[-self.predict_size:]['Close'].values
                norm_x, _ = normalize(df.iloc[-self.lookback:], df.iloc[-self.predict_size:]['Close'])
                data.append(norm_x)
                prices.append(actual_x)

        data = np.array(data)
        preds = self.model_tf.predict(data)

        for symbol, pred, current, price in zip(self.symbols.keys(), preds, data, prices):
            print(symbol, pred)
            self.place_order(symbol, pred, current[0], price)

        if sum([True if values['entered_trade'] else False for symbol, values in self.symbols.items()]) == 0:
            self.delay = 30

        if self.run_update_symbols:
            self.update_symbols()

        now = datetime.datetime.now(tz)

        if now.hour >= 15 or now.hour < 9:
            if now.hour == 15 and now.minute > 50:
                self.delay = 60*60
                print(f'Market end. Selling all shares.')
                for order in self.wb.get_account()['positions']:
                    symbol = order['ticker']['symbol']
                    if self.symbols[symbol]['entered_trade']:
                        order = self.wb.place_order(stock=symbol, action='SELL', orderType='MKT', enforce='DAY', quant=1)
                        print(f"SOLD: {symbol} --- Order#: {order.get('orderId')} --- EOD")
                        self.symbols[symbol]['entered_trade'] = False
            elif now.hour == 15 and now.minute > 29:
                self.delay = 5
                print(f'Market is about to close for the day. Making delay {self.delay} seconds.')

            elif now.hour == 8:
                self.delay = (now.replace(hour=9, minute=20, second=0) + datetime.timedelta(hours=24) - now).seconds
                print(f'Market is closed for the day. Sleeping for {self.delay/60/60} hours.')
            else:
                self.delay = 60*60
                print(f'Market is closed for the day. Sleeping for {1} hour.')

        self.s.enter(self.delay, 1, self.run)

    def get_data(self):
        try:
            self.s.run()
        except KeyboardInterrupt:
            print('Interrupted! Saving data.')
            for symbol in self.symbols.keys():
                self.save_data(symbol)
            pass


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.cuda.is_available())
    bot = WebullBot(creds=json.load(open('creds.json')), model_file='./dukascopy/model_res50_2021-03-04.pth')
    bot.get_data()
    tp.shutdown()
