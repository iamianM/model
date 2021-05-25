import mplfinance as fplt
from scipy.signal import find_peaks


def draw_chart(symbol, hist):
    mc = fplt.make_marketcolors(
        up='tab:blue', down='tab:red',
        edge='black',
        wick={'up': 'blue', 'down': 'red'},
        volume='green', alpha=1.0
    )
    s = fplt.make_mpf_style(marketcolors=mc, mavcolors=["yellow", "orange", 'brown'])
    image_file = f"./live_graphs/{symbol}_{hist.index[-1].isoformat().replace(':', '-')}.png"
    fplt.plot(
        hist.iloc[-46:],
        type='candle',
        style=s,
        volume=True,
        axisoff=True,
        mav=(5, 8, 13),
        figsize=(3.2, 3.2),
        xlim=(12.5, 44.5),
        savefig=image_file,
        tight_layout=True,
        returnfig=False
    )
    return image_file


def find_peaks_and_valleys(df, peak_width, peak_prominence, valley_width, valley_prominence):
    peaks, peak_prop = find_peaks(df['Close'], width=peak_width, prominence=peak_prominence)
    valleys, valley_prop = find_peaks(df['Close'] * (-1), width=valley_width, prominence=valley_prominence)
    return peaks, valleys


def grid_search_peaks_and_valleys(peak_valley_max_dist=32):
    widths = [None, 1, 2.5, 5, 10, 20]
    prominences = [None] + [(p, None) for p in [1, 2, 4, 6]]
    grid_search_values = {'peak': {'width': widths, 'prominence': prominences},
                          'valley': {'width': widths, 'prominence': prominences}}
    num_iters = 1
    for v1 in grid_search_values.values():
        for v2 in v1.values():
            num_iters *= len(v2)
    print(f'Running grid search on peaks and valleys. {num_iters} iterations.')

    grid_search = []
    for peak_width in grid_search_values['peak']['width']:
        for peak_prominence in grid_search_values['peak']['prominence']:
            for valley_width in grid_search_values['valley']['width']:
                for valley_prominence in grid_search_values['valley']['prominence']:
                    profits_p = []
                    profits = []
                    for file in files:
                        peaks, valleys = find_peaks_and_valleys(dfs[file], peak_width, peak_prominence, valley_width,
                                                                valley_prominence)

                        j = 0
                        for i in range(len(peaks)):
                            if j == len(valleys):
                                break
                            if valleys[j] >= peaks[i]:
                                continue
                            while j + 1 < len(valleys) and valleys[j + 1] < peaks[i]:
                                j += 1

                            if peaks[i] - valleys[j] < peak_valley_max_dist:
                                bought = dfs[file].iloc[valleys[j]]['Close']
                                sold = dfs[file].iloc[peaks[i]]['Close']
                                profits_p.append((sold - bought) / bought)
                                profits.append((sold - bought))

                            j += 1
                    if len(profits) > 0:
                        grid_search.append([sum(profits), sum(profits_p) / len(profits_p), len(profits), peak_width,
                                            peak_prominence, valley_width, valley_prominence])


def get_labels(df, peaks, valleys):
    labels = []
    for idx, local_time in enumerate(df['Local time']):
        local_time = local_time.split(' ')[1]
        if idx in peaks and '07:15:00' <= local_time <= '13:00:00':
            labels.append('Sell')
        elif idx in valleys and '07:15:00' <= local_time <= '12:00:00':
            labels.append('Buy')
        else:
            labels.append('Hold')

