import logging
from termcolor import colored

class Formatter(logging.Formatter):
    def __init__(self,verbose,debug) -> None:
        super(Formatter, self).__init__() # About super https://stackoverflow.com/questions/8853966/the-inheritance-of-attributes-using-init
        self.infocolors = {
            "DEBUG":    'green',
            "INFO":     'white',
            "WARNING":  'yellow' ,
            "ERROR":    'red' ,
            "CRITICAL": 'red'
        }
        self.infoattrs = {
            "DEBUG": [] ,
            "INFO":  [] ,
            "WARNING":[] ,
            "ERROR":  [] ,
            "CRITICAL": ['bold']
        }
        header = "{asctime} - {levelname:<8}" if verbose else "{levelname:<8}" 
        prefix = colored("{filename}:{name}:{lineno:<4} ",attrs=['dark']) if debug else (colored("{filename} ",attrs=['dark']) if verbose else "")
        format = "{message}"

        self.FORMATS = {
            logging.DEBUG:    self.get_header(header,"DEBUG")  +prefix + format,
            logging.INFO:     self.get_header(header,"INFO")   +prefix+ format,
            logging.WARNING:  self.get_header(header,"WARNING")    +prefix+ format,
            logging.ERROR:    self.get_header(header,"ERROR")  +prefix + format,
            logging.CRITICAL: self.get_header(header,"CRITICAL")   +prefix+ format
        }


    def get_header(self, msg, type="INFO"): 
        return f"[{colored(msg,self.infocolors[type],attrs=self.infoattrs[type])}] "
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,style='{',datefmt='%H:%M:%S')

        return formatter.format(record)


class Loggir(logging.Logger):
    def __init__(self,level=logging.DEBUG,verbose=False,debug=False) -> None:
        super(Loggir, self).__init__(__name__)
        self.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(Formatter(verbose,debug))
        self.addHandler(ch)