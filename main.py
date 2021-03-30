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
from keras.callbacks import EarlyStopping
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8,
              "September": 9, "October": 10, "November": 11, "December": 12}
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

def clean_frame(frame, drop_value):
    for e in frame.columns.to_list():
        if e != "date":
            for i in drop_value:
                list = frame[e].to_list()
                new_list = []
                for j in list:
                    new_list.append((str(j).replace(i, "")))
                frame[e] = new_list
    return frame

def compact_list(list):
    ret_list = []
    for e in list:
        for i in e:
            ret_list.append(i)
    return ret_list

def round_list(list, dec_points):
    ret_list = []
    for e in list:
        ret_list.append(round(e, dec_points))
    return ret_list


def dump_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def open_json(filename):
    with open(filename) as infile:
        data = json.load(infile)
    return data


class data_formatter():
    def __init__(self):
        self.format = self._format()

    class _format():
        def __init__(self):
            self.selected_features = []

        def _format_hyphen_datetimes(self, date_list):
            new_date_list = []
            for e in date_list:
                dates = e.split("-")
                date = datetime(int(dates[0]), int(dates[1]), int(dates[2]))
                new_date_list.append(date)
            return new_date_list

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
            self.bond_yield_df = full_df

        def pre_yield(self):
            for col in self.bond_yield_df.columns.to_list():
                if col != "date":
                    pre_col = self.bond_yield_df[col]
                    pre_col.index += 1
                    print(pre_col)
                    self.bond_yield_df["pre_" + col] = pre_col
            return self.bond_yield_df

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
            df = self._format_networth(df)
            self.selected_features.append(df)

        def discount_rate(self):
            df = pd.read_csv("data/dicount_rate.csv")
            df.columns = ["date", "discount_date"]
            df.to_csv("data/discount_rate.csv", index=False)
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def median_income(self):
            df = pd.read_csv("data/income.csv")
            df.columns = ["date", "median_income"]
            df.to_csv("data/income.csv", index=False)
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
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

        def CPI(self):
            df = pd.read_csv("data/CPI.csv")
            df.columns = ["date", "CPI"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def inflation(self):
            df = pd.read_csv("data/inflation.csv")
            df.columns = ["date", "inflation"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def eff_fed_funds_rate(self):
            df = pd.read_csv("data/eff_federal_funds_rate.csv")
            df.columns = ["date", "eff_fed_funds_rate"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def CSI(self):
            df = pd.read_csv("data/CSI.csv")
            df.columns = ["month", "year", "CSI"]
            month_list = df["month"].to_list()
            year_list = df["year"].to_list()
            new_date_list = []
            for e in range(len(year_list)):
                date = datetime(int(year_list[e]), int(month_dict[month_list[e]]), 1)
                new_date_list.append(date)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            df = df.drop("month", axis=1)
            df = df.drop("year", axis=1)
            self.selected_features.append(df)

        def outstanding_con_credit(self):
            df = pd.read_csv("data/outstanding_credit.csv")
            df.columns = ["date", "outstanding_consumer_credit"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def financial_literacy(self):
            df = pd.read_csv("data/financial_literacy.csv")
            df.columns = ["date", "fin_lit_18-34", "fin_lit_35-54", "fin_lit_55+"]
            date_list = df["date"].to_list()
            new_date_list = []
            for e in date_list:
                date = datetime(int(e), 1, 1)
                new_date_list.append(date)
            df["date"] = new_date_list
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def age_distribution(self):
            df = pd.read_csv("data/age_distribution.csv")
            df.columns = ["date", "0-14", "15-64", "65+"]
            date_list = df["date"].to_list()
            new_date_list = []
            for e in date_list:
                date = datetime(int(e), 1, 1)
                new_date_list.append(date)
            df["date"] = new_date_list
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def sp500(self):
            df = pd.read_csv("data/sp500.csv")
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            for e in df.columns.to_list():
                if e != "Close" and e != "date":
                    df = df.drop(e, axis=1)
            df.columns = ["date", "sp500"]
            self.selected_features.append(df)

        def VSTMX(self):
            df = pd.read_csv("data/VTSMX.csv")
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            for e in df.columns.to_list():
                if e != "Close" and e != "date":
                    df = df.drop(e, axis=1)
            df.columns = ["date", "VSTMX"]
            self.selected_features.append(df)

        def goverment_rev(self):
            df = pd.read_csv("data/government_rev.csv")
            df.columns = ["date", "government_rev"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def goverment_expend(self):
            df = pd.read_csv("data/government_expend.csv")
            df.columns = ["date", "government_expend"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def fed_bond_holdings(self):
            df = pd.read_csv("data/fed_bond_holdings.csv")
            df.columns = ["date", "fed_bond_holdings"]
            date_list = df["date"].to_list()
            new_date_list = self._format_hyphen_datetimes(date_list)
            df["date"] = new_date_list
            df = df.reset_index(drop=True)
            self.selected_features.append(df)

        def GDP_growth(self):
            df = pd.read_csv("data/GDP.csv")
            df = df[df["Country Name"] == "United States"]
            df = df.iloc[:, 5:len(df.columns) - 2]
            years = df.columns.to_list()
            values = df[0:1].values.tolist()
            new_date_list = []
            for e in years:
                date = datetime(int(e), 1, 1)
                new_date_list.append(date)
            df = pd.DataFrame({"date": new_date_list, "GDP_growth": values[0]})
            self.selected_features.append(df)

    def compile(self):
        org_frame = self.format.bond_yield_df
        for comp_frame in self.format.selected_features:
            list_dict = {}
            for i in comp_frame.columns.to_list():
                list_dict[i] = []
            current_index = 0

            for e in range(1,len(comp_frame)):
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

class analyzer():
    def __init__(self, model, frame, search_col):
        self.model = model
        self.frame = frame
        self.search_col = search_col
        self.true = frame[search_col]

    def set_data_groups(self, split):
        self.training_features, self.training_labels, \
        self.test_features, self.test_labels, self.training_dates, self.test_dates = prepare_training_data(self.frame, split, self.search_col)
        self.get_prediction()
        self.compact_predictions()

    def get_prediction(self):
        self.training_pred = round_list(compact_list(self.model.predict(self.training_features)),2)
        self.test_pred = round_list(compact_list(self.model.predict(self.test_features)),2)

    def compact_predictions(self):
        full_pred_list = []
        full_label_list = []

        full_pred_list.append(self.training_pred)
        full_pred_list.append(self.test_pred)

        full_label_list.append(self.training_labels)
        full_label_list.append(self.test_labels)

        self.full_true = compact_list(full_label_list)
        self.full_pred = compact_list(full_pred_list)

    def plot_prediction(self, pred, true, dates):
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=75))
        plt.plot(dates, pred, label="Forudsigelse")
        plt.plot(dates, true, label="Reelle værdier")
        plt.xlabel("Dato")
        plt.ylabel("Direkte rente")
        plt.gcf().autofmt_xdate()
        plt.title(f"Forudsigelse og reelle værdier \n for US. Treasuries Bonds med løbetid på 10 år \n for test data i perioden {dates[0]} til {dates[len(dates)-1]}")
        plt.legend()
        plt.show()

    def plot_history(self, history):
        acc = history['mae']
        val_acc = history['val_mae']
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(acc) + 1)
        """plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()"""

        plt.plot(epochs, loss, label='Trænings tab(MAE)')
        plt.plot(epochs, val_loss, label='Test tab(MAE)')
        plt.title('Oversigt over tab under træning, \n for den direkte rente på US Treasuries med løbetid på 10 år')
        plt.xlabel("Epoch")
        plt.ylabel("Tab(MAE)")
        plt.legend()

        plt.show()

    def _calc_residiuals(self, pred, true):
        dif_list = []
        for e in range(len(true)):
            dif_list.append(abs(true[e] - pred[e]))
        return dif_list

    def calc_rsquared(self,pred, true):
        res = self._calc_residiuals(true, pred)
        res_list = []
        for e in res:
            res_list.append(e ** 2)

        avg = sum(true) / len(true)
        avg_dif = []
        for e in true:
            avg_dif.append((e - avg) ** 2)

        lower = sum(avg_dif)
        upper = sum(res_list)

        rsquared = 1 - (upper/lower)
        print("Modellen for obligationer med løbetid på 20 år")
        print("R2:", rsquared)

    def calc_rsquared_adjusted(self,pred, true):
        K = len(self.frame.columns.to_list())-1
        n = len(self.frame)
        res = self._calc_residiuals(true, pred)
        res_list = []
        for e in res:
            res_list.append(e ** 2)

        avg = sum(true) / len(true)
        avg_dif = []
        for e in true:
            avg_dif.append((e - avg) ** 2)

        lower = sum(avg_dif)
        upper = sum(res_list)

        rsquared_adjusted = 1 - ((upper/(n-K))/(lower/(n-1)))
        print("R2 justeret:", rsquared_adjusted)

    def calc_direc_succes_rate(self, pred, true):
        base = true[0:len(true) - 1]
        true = true[1:len(true)]
        pred = pred[1:len(pred)]
        down_correct = []
        up_correct = []
        neutral_correct = []
        neutral_wrong = []
        down_wrong = []
        up_wrong = []
        for e in range(len(base)):
            if pred[e] > base[e]:
                if true[e] > base[e]:
                    up_correct.append(e)
                else:
                    up_wrong.append(e)
            elif pred[e] < base[e]:
                if true[e] < base[e]:
                    down_correct.append(e)
                else:
                    down_wrong.append(e)
            elif pred[e] == base[e]:
                if true[e] == base[e]:
                    neutral_correct.append(e)
                else:
                    neutral_wrong.append(e)
        print("Fordeling af forudset retning \n for modellen for obligationer \n med løbetid på 10 år")
        print("Antal")
        num_df = pd.DataFrame({"Retning": ["Op", "Samme", "Ned", "Sum"],
                           "Korrekt": [len(up_correct), len(neutral_correct), len(down_correct), len(up_correct)+len(neutral_correct)+len(down_correct)],
                           "Forkert": [len(up_wrong), len(neutral_wrong), len(down_wrong), len(up_wrong) + len(neutral_wrong) + len(down_wrong)]},
                            )
        print(num_df.to_string())
        print("Procent")
        #+1 -1 tilføjet for at forhindre dividering med 0
        dec_point = 2
        pct_df = pd.DataFrame({"Retning": ["Op", "Samme", "Ned", "Sum"],
                           "Korrekt": [round(len(up_correct)/(len(up_correct)+len(up_wrong))*100, dec_point), round(((len(neutral_correct)+1)/(len(neutral_correct)+len(neutral_correct)+1)-1)*100, dec_point), round(len(down_correct)/(len(down_wrong)+len(down_correct))*100, dec_point),
                                       round((len(up_correct) + len(neutral_correct) + len(down_correct))/(len(up_correct) + len(neutral_correct) + len(down_correct)+ len(up_wrong) + len(neutral_wrong) + len(down_wrong))*100,dec_point)],
                           "Forkert": [round((len(up_wrong)/(len(up_correct)+len(up_wrong))*100), dec_point), round((((len(neutral_wrong)+1)/(len(neutral_correct)+len(neutral_correct)+1)-1)*100),dec_point), round(len(down_wrong)/(len(down_wrong)+len(down_correct))*100,dec_point),
                                       round(((len(up_wrong) + len(neutral_wrong) + len(down_wrong))/((len(up_correct) + len(neutral_correct) + len(down_correct)+ len(up_wrong) + len(neutral_wrong) + len(down_wrong)))*100), dec_point)]},
                          )
        print(pct_df.to_string())
        return df


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
    return df["pred_dif"].mean()

def prepare_training_data(frame, test_split, test_col):

    frame = frame.reset_index(drop=True)
    labels = []
    for e in range(len(frame)-1):
        labels.append(float(frame.at[e+1, test_col]))
    training_features = frame[0: int(len(frame)*(1-test_split))]
    test_features = frame[int(len(frame)*(1-test_split)): len(frame)-1]
    training_labels = labels[0: int(len(frame)*(1-test_split))]
    training_dates = training_features["date"].to_list()
    test_dates = test_features["date"].to_list()
    training_features = training_features.drop("date", axis=1)
    test_features = test_features.drop("date", axis=1)
    test_labels = labels[int(len(frame)*(1-test_split)): len(frame)-1]
    scaler = MinMaxScaler().fit(training_features)
    return scaler.transform(training_features), np.array(training_labels), \
           scaler.transform(test_features), np.array(test_labels), training_dates, test_dates

def train_model(frame, test_split, test_col):
    training_features, training_labels, test_features, test_labels = prepare_training_data(frame, test_split, test_col)
    model = Sequential([
        Dense(20, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='relu')
    ])
    es = EarlyStopping(monitor='val_mae', verbose=1, patience=5)
    model.compile(optimizer='sgd', loss='mae', metrics=['mae'], )
    hist = model.fit(training_features, training_labels, batch_size=16, epochs=25,
                     validation_data=(test_features, test_labels))
    model.save("model")
    dump_json("hist", hist.history)
    res = evaluate_model(model, test_features, test_labels)
    return model, hist.history


def compile_new_frame(years):
    data = data_formatter()
    data.format.bond_yield(years)
    data.format.median_networth()
    data.format.mean_networth()
    data.format.discount_rate()
    data.format.avg_income()
    data.format.median_income()
    data.format.reserve_requirements()
    data.format.CPI()
    data.format.inflation()
    data.format.outstanding_con_credit()
    data.format.financial_literacy()
    data.format.age_distribution()
    data.format.sp500()
    data.format.VSTMX()
    data.format.goverment_rev()
    data.format.goverment_expend()
    data.format.fed_bond_holdings()
    data.format.GDP_growth()
    return data.compile()

def save(frame):
    frame.to_csv("compiled_frame", index=False)

def create_new_frame(year):
    df = compile_new_frame(year)
    df = clean_frame(df, ["$", ","])
    save(df)
    return df


