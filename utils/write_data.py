import os
class WriteData():
    def __init__(self,root_path):
        self.root_path = root_path
    def write_data_txt(self,content):
        f = open(self.root_path,'a')
        f.write(content)      
        f.write('\n')
        f.close()     

if __name__=='__main__':
    root_path = 'debug.log'
    files = WriteData(root_path)
    files.write_data_txt('love')
    files.write_data_txt('love2')