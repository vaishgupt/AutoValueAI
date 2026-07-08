import sys

def error_message(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    return f"""
Error occurred in python script

File Name : {exc_tb.tb_frame.f_code.co_filename}

Line Number : {exc_tb.tb_lineno}

Error : {str(error)}
"""