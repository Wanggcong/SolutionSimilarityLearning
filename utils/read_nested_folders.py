'''
root_path
--A
----a1
----a2
----a3
--B
----b1
----b2

return [[a1,a2,a3],[b1,b2]]
'''
import os
class read_folders():
    def __init__(self,root_path):
        self.root_path = root_path
        self.nested_files = []
    def get_nested_folders(self):
        first_subfolders = sorted(os.listdir(self.root_path))
        for i in range(len(first_subfolders)):
            one_folder_path = os.path.join(self.root_path,first_subfolders[i])    
            one_folder = sorted(os.listdir(one_folder_path))
            one_folder_abs = []
            for j in range(len(one_folder)):
                one_folder_abs.append(os.path.join(one_folder_path,one_folder[j]))
            self.nested_files.append(one_folder_abs)


if __name__=='__main__':
    root_path = './data'
    files = read_folders(root_path)
    files.get_nested_folders()
    print('files',files.nested_files)