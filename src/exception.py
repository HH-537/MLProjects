import sys


def construct_error_message(err, err_detail: sys):
    _, _, exec_traceback = err_detail.exc_info()
    error_message = 'Error occurred in script name [{0}], in line number [{1}]. Error is "[{2}]'.format(
        exec_traceback.tb_frame.f_code.co_filename,
        exec_traceback.tb_frame.f_lineno,
        str(err)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_message = construct_error_message(error_msg, error_detail)

    def __str__(self):
        return self.error_message
