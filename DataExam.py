import Calendar as calendar
import pandas as pd

#----------------------------------------------------------------------------
# TODO
def basic_summary(filename, data_path = "Data/"):

    print("Reading sales data...")
    sales = pd.read_csv(data_path + "sales_train_validation.csv")

    print("Reading calendar...")
    cal = calendar.cCalendar(data_path)

    results = pd.DataFrame()
    results['id'] = sales['id']
    results.set_index('id')

    d_cols = [c for c in sales.columns if 'd_' in c]
    summary = create_summary(sales, d_cols, "full_")
    results = pd.concat([results, summary], axis=1, sort=False)

    years = [2011, 2012, 2013, 2014, 2015, 2016]
    for year in years:
        print("\n\nLooking for year " + str(year))
        d_cols = cal.get_dcol_for_year(year)
        summary = create_summary(sales, d_cols, str(year) + "_")
        results = pd.concat([results, summary], axis=1, sort=False)

    results.to_csv(filename)


#----------------------------------------------------------------------------
# TODO
def create_summary(data, d_cols, prefix = ""):        
    results = pd.DataFrame()
    print(len(d_cols))
    results[prefix + 'count'] = str(len(d_cols))
    #results[prefix + 'min'] = data[d_cols].min(axis=1)
    #results[prefix + 'max'] = data[d_cols].max(axis=1)
    #results[prefix + 'sum'] = data[d_cols].sum(axis=1)
    #results[prefix + 'mean'] = data[d_cols].mean(axis=1)
    #results[prefix + 'median'] = data[d_cols].median(axis=1)
    #results[prefix + 'std'] = data[d_cols].std(axis=1)
    results[prefix + 'zeros'] = (data[d_cols] == 0).astype(int).sum(axis=1)
    results[prefix + 'pcnt_zeros'] = ((data[d_cols] == 0).astype(int).sum(axis=1))/len(d_cols)
    return results


#----------------------------------------------------------------------------
# TODO
def week_summary(self, filename, days=['Monday']):
    results = pd.DataFrame()
    results['id'] = self.sales['id']

    for day in days:
        d_cols = self.calendar.getDays(day)
        subset = self.sales[self.limit_d_cols(d_cols)]
            
        name = "default"
        if(isinstance(day,str)):
            name = day 
        elif(isinstance(day,list)):
            name = "_".join(str(x) for x in day)

        summary = self.create_summary(subset, name + "_")
        results = pd.concat([results, summary], axis=1, sort=False)

    results.set_index('id')
    results.to_csv(filename)