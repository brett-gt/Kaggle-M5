from abc import ABCMeta, abstractmethod
from sklearn import preprocessing

#--------------------------------------------------------------------------------
# Calendar.csv All Columns
# [date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type 1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI]
#
# sales_train_validation.csv
# This is arranged where every row is an item and most columns are how many sold on a particular day.
# [id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, d_3, ... , d_1913]  
#
# sample_submission.csv
# [id, F1, F2, F3, ...., F28]
#
# sell_price.csv
# Rows are the prices over time for different items
# [store_id, item_id, wm_yr_wk, sell_price]
#



#--------------------------------------------------------------------------------
class cDataEncoding(metaclass=ABCMeta):
    """ Abstract class that defines parmaters require to parse raw data into desired encoding schemes
    """
    BOOL_DICT  = {'True': True, 'False': False} 


    def __init__(self):
        pass

    @property
    @abstractmethod
    def FILENAME(self):
        pass

    @property
    @abstractmethod
    def COL_TO_ONEHOT(self):
        pass

    @property
    @abstractmethod
    def COL_NO_FORMAT(self):
        pass

    @property
    @abstractmethod
    def NORM_METHOD(self):
        pass

    @property
    @abstractmethod
    def COL_TO_NORM(self):
        pass

    @property
    @abstractmethod
    def COL_FOR_SCORING(self):
        pass

    @property
    @abstractmethod
    def LABEL_MAPPING(self):
        pass

    @property
    @abstractmethod
    def COL_TO_USE(self):
        pass

#--------------------------------------------------------------------------------
class cAutoEncoding(cDataEncoding):
    FILENAME = "combined_test_data.pkl"

    COL_TO_ONEHOT = ["Protocol"]

    COL_NO_FORMAT = []

    NORM_METHOD = preprocessing.MinMaxScaler()

    COL_TO_NORM =   ["Destination Port", "Flow Duration", "Total Fwd Packets","Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                     "Fwd Packet Length Mean",	"Fwd Packet Length Std", 
                     "Fwd IAT Mean", "Fwd IAT Std", "Flow IAT Max",	"Flow IAT Min", "Bwd IAT Mean","Bwd IAT Std", "Packet Length Mean","Packet Length Std",
                     "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", 
                     "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
                     "Average Packet Size", "Avg Fwd Segment Size", 
                     "Subflow Fwd Packets", "Subflow Fwd Bytes",  
                     "Init_Win_bytes_forward", "min_seg_size_forward",
                     "Active Mean", "Active Std", "Active Max", "Active Min"]

    COL_FOR_SCORING = ["Label", "Timestamp", "Full Label"]

    LABEL_MAPPING = {'BENIGN': False, 'Bot': True, 'DDoS': True, 'DoS GoldenEye': True, 'DoS Hulk': True, 'DoS Slowhttptest': True, 
                     'DoS slowloris': True, 'FTP-Patator': True, 'Heartbleed': True, 'Infiltration': True,
                     'PortScan': True, 'SSH-Patator': True, "Web Attack \x96 Brute Force": True, "Web Attack \x96 Sql Injection": True, "Web Attack \x96 XSS": True}

    COL_TO_USE = COL_TO_ONEHOT + COL_TO_NORM + COL_FOR_SCORING
