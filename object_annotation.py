# this is just a placeholder file.  somehow it is required to unpickle inria annotations

class ObjectAnnotation(object):
    def __init__(self, file_name, bb_location, bb_size):
        self.file_name = file_name
        self.bb_location = bb_location
        self.bb_size = bb_size
