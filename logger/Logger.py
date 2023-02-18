
class Logger:

    mode = 3

    @classmethod
    def action(cls, logging_mode, key, message):
        logging_mode=logging_mode+':'
        rem = 10 - len(logging_mode)
        for i in range(rem):
            logging_mode=logging_mode + ' '
        
        key+=":"
        rem = 15 - len(key)
        for i in range(rem):
            key=key + ' '
        
        print(logging_mode+key+message)
    
    @classmethod
    def d(cls, key, message):
        if cls.mode>=4:
            cls.action('Debug', key, message)

    @classmethod
    def e(cls, key, message):
        cls.action('Error', key, message)

    @classmethod
    def i(cls, key, message):
        if cls.mode>=3:
            cls.action('Info', key, message)
    
    @classmethod
    def w(cls, key, message):
        if cls.mode>=2:
            cls.action('Warning', key, message)
