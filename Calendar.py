import pandas as pd

class cCalendar:
    """description of class"""
    data = []

    #----------------------------------------------------------------------------
    def __init__(self, path):
        print("\ncCalendar: initializing...")
        print("Reading calendar.csv...")
        self.data = pd.read_csv(path + "calendar.csv")

    #----------------------------------------------------------------------------
    def wm_yr_wk_to_dcol(self, row):
        """Take in the wm_yr_wk format and return the d_col entry
        """
        result = self.data.loc[self.data['wm_yr_wk']==row.wm_yr_wk].d
        if(result.empty):
            return None
        else:
            return result.values

    #---------------------------------------------------------------------------
    def getDays(self, days, limit="d_1963"):
        """ Takes in an either a string ("Monday") or list of day names (["Monday", "Tuesday", etc]) 
            and returns a d_XXXX list corresponding to that day.

            TODO: May need to implement limit to stop for looking past end of dataset.

            TODO: May be overcomplicating it with allowing list or string.
        """
        print("\ncCalendar: getDays " + str(days))

        if(isinstance(days,str)):
            result = self.data.loc[self.data['weekday'] == days].d
        elif(isinstance(days,list)):
            result = self.data.loc[self.data['weekday'].isin(days)].d

        if(result.empty):
            return None
        else:
            return result.values

    #--------------------------------------------------------------------------------
    def get_dcol_for_year(self, year):
        """ Gets the dcol range for a particular year
        """
        start_date = str(year) + "-01-01"
        end_date = str(year) + "-12-31"

        return self.get_date_range(start_date, end_date)


    #--------------------------------------------------------------------------------
    def get_date_range(self, date_start, date_end):
        """ Get a range of rows from a dataset using the d_XXXX value.

            Arguments:
                data- dataset that contains a "d" column (d_XXXX)
                d_start - first d_XXXX
                d_end   - end d_XXXX

            Returns:
                data frame
        """  
        d_start = self.get_d_from_date(date_start)
        if(d_start == None):
            d_start = "d_1"
        d_end = self.get_d_from_date(date_end)
        if(d_end == None):
            d_end = "d_1913"
        return self.get_d_range(d_start, d_end)

    #--------------------------------------------------------------------------------
    def get_d_range(self, d_start, d_end):
        """ Get a range of rows from a dataset using the d_XXXX value.
        """  
        d_start_num = int(d_start.split('_')[1])
        d_end_num = int(d_end.split('_')[1]) + 1

        print(d_start_num)
        print(d_end_num)

        result = []

        for x in range(d_start_num, d_end_num):
            result.append("d_" + str(x))
        return result


    #--------------------------------------------------------------------------------
    def get_d_from_date(self, date):
        """ The common date identified is d_XXXX where XXXX is a numbered list from
            1 to 1969 (currently).  Use this function to get the d_XXXX that corresponds
            to a specific date.  Uses calendar dataframe (loaded from calendar.csv) for
            lookup.

            Arguments:
                date - (datetime) format year-month-day with 0 padding

            Returns:
                string - d_XXXX or null if not found
        """  
        result = self.data.loc[self.data['date']==date].d
        if(result.empty):
            return None
        else:
            return result.values[0]

    #--------------------------------------------------------------------------------
    def get_date_from_d(d):
        """ The common date identified is d_XXXX where XXXX is a numbered list from
            1 to 1969 (currently).  Use this function to get the date associated with a
            d_XXXX  Uses calendar dataframe (loaded from calendar.csv) for
            lookup.

            Arguments:
                calendar - calendar dataframe (pre-loaded from calendar.csv)
                d - d_XXXX 

            Returns:
                datetime - format year-month-day with 0 padding
        """ 
        result = self.data.loc[calendar['d']==d].date
        if(result.empty):
            return None
        else:
            return datetime.datetime.strptime(result.values[0],globals.DATE_FORMAT)
