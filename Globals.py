
YEARS = [2011, 2012, 2013, 2014, 2015]

DATE_FORMAT = "%Y-%m-%d"

DAYS_OF_WEEK = ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"]
WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
WEEKEND = ["Saturday","Sunday"]

WEEK_SEARCH = ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday", WEEKDAYS, WEEKEND]


D_START_VAL = "d_1914"
D_END_VAL = "d_1941"
D_START_EVAL = "d_1942"
D_END_EVAL = "d_1969"


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