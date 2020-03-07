import pandas as pd

class cSalesData:
    """description of class"""

    data = []

    #----------------------------------------------------------------------------
    def __init__(self, path):
        self.data = pd.read_csv(path + "sales_train_validation.csv")

    #----------------------------------------------------------------------------
    def save_summary(self, filename):
        d_cols = [c for c in self.data.columns if 'd_' in c]

        results = pd.DataFrame()
        results['id'] = self.data['id']
        results.set_index('id')
        results['count'] = len(d_cols)
        results['min'] = self.data[d_cols].min(axis=1)
        results['max'] = self.data[d_cols].max(axis=1)
        results['sum'] = self.data[d_cols].sum(axis=1)
        results['mean'] = self.data[d_cols].mean(axis=1)
        results['median'] = self.data[d_cols].median(axis=1)
        results['std'] = self.data[d_cols].std(axis=1)
        results['zeros'] = (self.data[d_cols] == 0).astype(int).sum(axis=1)

        results.to_csv(filename)








