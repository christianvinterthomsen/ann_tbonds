import urllib.request
from pprint import pprint
from html_table_parser import HTMLTableParser
import pandas as pd
from datetime import datetime
import tensorflow
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np

month_dict = {"January": 1, "December": 12}
def url_get_contents(url):
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    return f.read()

def parse_html(url):
    xhtml = url_get_contents(url).decode('utf-8')
    p = HTMLTableParser()
    p.feed(xhtml)
    return p

def create_table(url, index):
    p = parse_html(url)
    df = pd.DataFrame(p.tables[0])
    return df


class data_formatter():
    def __init__(self):
        self.format = self._format()
    class _format():
        def __init__(self):
            self.selected_features = []

        def _single_yield(self, year):
            url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=" + year
            df = create_table(url, 0)
            col_names = df[0:1].values.tolist()[0]
            df = df.drop(0, axis=0)
            col_names.pop(0)
            col_names.insert(0, "date")
            df.columns = col_names
            date_list = df["date"].to_list()
            new_date_list = []
            for e in date_list:
                dates = e.split("/")
                date = datetime(int("20" + dates[2]), int(dates[0]), int(dates[1]))
                new_date_list.append(date)
            df["date"] = new_date_list
            return df

        def bond_yield(self, years):
            full_df = pd.DataFrame()
            for year in years:
                print("Extracting yeary:", year)
                ret_df = self._single_yield(year)
                full_df = full_df.append(ret_df)
            full_df = full_df.reset_index(drop=True)
            print(full_df)
            self.bond_yield = full_df

        def _format_networth(self, file):
            needed_col = ["date", "net_mean_worth", "net_median_worth"]
            for e in file.columns.to_list():
                if e not in needed_col:
                    file = file.drop(e, axis=1)
            new_date_list = []
            for e in file["date"].to_list():
                date = datetime(int(e), 1, 1)
                new_date_list.append(date)
            file["date"] = new_date_list
            file = file.reset_index(drop=True)
            return file

        def median_networth(self):
            df = pd.read_csv("data/median_networth.csv")
            df = self._format_networth(df)
            self.selected_features.append(df)

        def mean_networth(self):
            df = pd.read_csv("data/mean_networth.csv")
            df = self.format_networth(df)
            self.selected_features.append(df)

        def discount_rate(self):
            df = pd.read_csv("data/dicount_rate.csv")
            df.columns = ["date", "discount_date"]
            df.to_csv("data/discount_rate.csv", index=False)
            date_list = df["date"].to_list()
            new_date_list = []
            for e in date_list:
                dates = e.split("-")
                date = datetime(int(dates[0]), int(dates[1]), int(dates[2]))
                new_date_list.append(date)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def median_income(self):
            df = pd.read_csv("data/income.csv")
            df.columns = ["date", "median_income"]
            df.to_csv("data/income.csv", index=False)
            date_list = df["date"].to_list()
            new_date_list = []
            for e in date_list:
                dates = e.split("-")
                date = datetime(int(dates[0]), int(dates[1]), int(dates[2]))
                new_date_list.append(date)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def avg_income(self):
            url = "https://dqydj.com/household-income-by-year/"
            df = create_table(url, 0)
            df = df.drop(0, axis=0)
            df.columns = ["date", "avg. income", "inflation adj.(2020)"]
            dates = df["date"].to_list()
            new_date_list = []
            for e in dates:
                date = datetime(int(e), 1, 1)
                new_date_list.append(date)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def reserve_requirements(self):
            p = parse_html("https://www.federalreserve.gov/monetarypolicy/reservereq.htm")
            df = pd.DataFrame(p.tables[1])
            df = df.drop(0, axis=0)
            df.columns = ["date", "low_reserve_trance", "exemption_amount"]
            date_list = df["date"].to_list()
            new_date_list = []
            for e in date_list:
                dates = e.split(" ")
                day = dates[1].split(",")[0]
                date = datetime(int(dates[2]), int(month_dict[dates[0]]), int(day))
                new_date_list.append(date)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

    def compile(self):
        org_frame = self.format.bond_yield
        for comp_frame in self.format.selected_features:
            print(comp_frame.to_string())
            list_dict = {}
            for i in comp_frame.columns.to_list():
                list_dict[i] = []
            current_index = 0

            for e in range(len(comp_frame)):
                if comp_frame.at[e, "date"] >= org_frame.at[0, "date"]:
                    current_index = e - 1
                    break
            for e in range(len(org_frame)):
                org_frame_date = org_frame.at[e, "date"]
                comp_cur_date = comp_frame.at[current_index, "date"]
                if current_index + 1 != len(comp_frame):
                    comp_next_date = comp_frame.at[current_index + 1, "date"]
                if org_frame_date >= comp_cur_date and org_frame_date < comp_next_date:
                    for i in list_dict:
                        list_dict[i].append(comp_frame.at[current_index, i])
                elif current_index + 1 < len(comp_frame):
                    for i in list_dict:
                        list_dict[i].append(comp_frame.at[current_index + 1, i])
                    current_index += 1
                else:
                    for i in list_dict:
                        list_dict[i].append(comp_frame.at[len(comp_frame) - 1, i])
            for e in list_dict:
                if e != "date":
                    org_frame[e] = list_dict[e]
        return org_frame


def evaluate_model(model, test_features, test_labels):
    pred = model.predict(test_features)
    dif_list = []
    change_list = []
    for e in range(len(test_labels)-1):
        dif_list.append(abs(test_labels[e]-pred[e]))
        change_list.append(abs(test_labels[e+1]-test_labels[e]))
    df = pd.DataFrame({"change": change_list, "pred_dif": dif_list})
    print("Average change:",df["change"].mean())
    print("Average residual:", df["pred_dif"].mean())

def prepare_training_data(frame, test_split, test_col):
    frame = frame.drop("date", axis=1)
    frame = frame.reset_index(drop=True)
    labels = []
    for e in range(len(frame)-1):
        labels.append(float(frame.at[e+1, test_col]))
    training_features = frame[0: int(len(frame)*(1-test_split))]
    test_features = frame[int(len(frame)*(1-test_split)): len(frame)-1]
    training_labels = labels[0: int(len(frame)*(1-test_split))]
    test_labels = labels[int(len(frame)*(1-test_split)): len(frame)-1]
    scaler = MinMaxScaler().fit(training_features)
    return scaler.transform(training_features), np.array(training_labels), \
           scaler.transform(test_features), np.array(test_labels)

def train_model(frame, test_split, test_col):
    training_features, training_labels, test_features, test_labels = prepare_training_data(frame, test_split, test_col)
    print(len(training_labels))
    model = Sequential([
        Dense(15, activation='relu', input_shape=[0,len(training_features[0])]),
        Dense(30, activation='relu'),
        Dense(100, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['acc'])
    model.fit(training_features, training_labels, batch_size=1, epochs=50)
    evaluate_model(model, test_features, test_labels)
    return model



"""years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008",
         "2009", "2010", "2011","2012","2013","2014","2015","2016","2017","2018",
         "2019","2020"]
#years = ["2020"]
price_df = get_multiple_years(years)
discount_df = format_discount_rate()
reservereq_df = format_reservereq_contents()
income_df = format_income()

full_frame = match_datetimes(price_df, discount_df)
full_frame = match_datetimes(full_frame, reservereq_df)
full_frame = match_datetimes(full_frame, income_df)

for c in full_frame.columns:
    full_frame[c] = pd.to_numeric(price_df[c], errors='coerce')
full_frame = full_frame.fillna(0)

model = train_model(full_frame, 0.1, "30 yr")"""
"""
discount_df = format_discount_rate()
print(discount_df.to_string())
reservereq_df = format_reservereq_contents()
print(reservereq_df)
print(reservereq_df.columns.to_list())

full_frame = match_datetimes(price_df, discount_df)
full_frame = match_datetimes(full_frame, reservereq_df)
print(full_frame.to_string())

income_df = format_income()
print(income_df)"""

year = ["2018", "2019", "2020"]

data = data_formatter()
data.format.bond_yield(year)
data.format.avg_income()
data.format.median_networth()
df = data.compile()
print(df.to_string())


