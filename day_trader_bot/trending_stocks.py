import requests
from webull import paper_webull
import datetime
import json
import pandas as pd
from os import listdir
import time
import sched

URL = "https://api.stocktwits.com/api/2/trending/symbols.json?access_token=c8246a247edb65d655d4e1cd3781df676b4a8f76"
PATH = './trending_stock_data'
s = sched.scheduler(time.time, time.sleep)
symbols = json.load(open('./trending_stocks.json'))
hists = {}
for file in [f for f in listdir(PATH) if f.endswith('.csv')]:
    hists[file] = pd.read_csv(f"{PATH}/{file}", index_col='Local time')


def login_webull(creds):
    wb = paper_webull()
    wb.login(creds['email'], creds['password'])
    return wb


creds = json.load(open('creds.json'))
wb = login_webull(creds)


def get_trending_stocks():
    r = requests.get(URL).json()
    return [rr['symbol'] for rr in r['symbols']]


def get_stock_data(wb, symbol, interval='m1', count=1000):
    new_day = datetime.time(9, 31)

    # for symbol in new_symbols:
    hist = wb.get_bars(stock=symbol, interval=interval, count=count, extendTrading=0)
    time_before = hist.index[0] - datetime.timedelta(seconds=60)
    good = True
    for time, row in hist.iterrows():
        if (time - time_before).seconds != 60 and time.time() != new_day:
            good = False
            print(symbol, 'no good')
            break
        time_before = time

    if good:
        return hist
    else:
        return None


def update_symbols():
    global symbols
    new_symbols = [symbol for symbol in get_trending_stocks() if symbol not in symbols]
    for symbol in new_symbols:
        symbols[symbol] = datetime.datetime.now().isoformat()
    print(new_symbols)


def update_hists():
    global hists
    now = datetime.datetime.now()

    new_hists = []
    for symbol, date in symbols.items():
        if symbol.endswith('.X') or symbol.endswith('.B') or symbol.endswith('.CA'):
            new_hists.append(symbol)
            continue
        date_t = datetime.datetime.fromisoformat(date)
        skip_days = 2
        if date_t.day >= 4:
            skip_days += 7 - date_t.day
        if now > date_t + datetime.timedelta(days=skip_days):
            print(symbol)
            try:
                hist = get_stock_data(wb, symbol)
            except:
                new_hists.append(symbol)
                continue

            if hist is not None:
                hist.columns = [c.capitalize() for c in hist.columns]
                hist.index.name = 'Local time'
                new_hists.append(symbol)
                file = f"{symbol}_{date.split('T')[0]}.csv"
                if symbol in hists:
                    hists[file] = pd.concat([hists[symbol], hist[[idx not in hists[file] for idx in hist.index]]])
                else:
                    hists[file] = hist

    for symbol in new_hists:
        del symbols[symbol]

    print(new_hists)


def run():
    global symbols
    global hists
    # print('symbols')
    # s.enter(10, 1, update_symbols)
    # print('hists')
    # s.enter(15, 1, update_hists)
    update_symbols()
    update_hists()

    json.dump(symbols, open('trending_stocks.json', 'w'))

    for file, hist in hists.items():
        hist.to_csv(f'./trending_stock_data/{file}')

    time.sleep(5*60)


def get_data():
    try:
        while True:
            run()
    except KeyboardInterrupt:
        print('Interrupted! Saving data.')
        json.dump(symbols, open('trending_stocks.json', 'w'))

        for file, hist in hists.items():
            hist.to_csv(f'./trending_stock_data/{file}')
    except Exception as e:
        print(e)


if __name__ == "__main__":
    get_data()
