import shutil

def move_folder(original,target):
    original = r''+original+''
    target =r''+target+'\\'+original

    shutil.move(original, target)