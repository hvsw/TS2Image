from datetime import datetime

__all__ = ["log"]

def log(msg:str, log_file: str = None):
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    txt = f'{now}: {msg}'
    if log_file is None:
        print(txt)
    else:
        f = open(log_file, 'w')
        f.write(txt)
        f.close