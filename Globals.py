
YEARS = [2011, 2012, 2013, 2014, 2015]

DATE_FORMAT = "%Y-%m-%d"


#--------------------------------------------------------------------------------
def d_to_num(d_col):
    return int(d_col.split('_')[1])

#--------------------------------------------------------------------------------
def num_to_d(num):
    return "d_" + str(num)

#--------------------------------------------------------------------------------
def make_d_col_range(start_d_col, end_d_col):
    d_cols = []
    for i in range(d_to_num(start_d_col),d_to_num(end_d_col)+1):
        d_cols.append(num_to_d(i))
    return d_cols