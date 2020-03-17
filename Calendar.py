import pandas as pd
import Globals as globals

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

        print("\ncCalendar: get_d_from_date: Looking for " + str(date) + " is " + str(result))

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

        print("\ncCalendar: get_date_from_d: Looking for " + str(d) + " is " + str(result))

        if(result.empty):
            return None
        else:
            return datetime.datetime.strptime(result.values[0],globals.DATE_FORMAT)
