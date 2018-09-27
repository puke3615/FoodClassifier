import os

POSTFIX = ['png', 'PNG', 'jpg', 'jpeg', 'JPG', 'JPEG']


def is_image(file):
    return any(map(lambda x: file.endswith(x), POSTFIX))


def get_all_files(path, file_selector=lambda filepath: True, dir_selector=lambda dirpath: True, result=None):
    result = [] if result is None else result
    if not os.path.exists(path):
        raise Exception('File "%s" not found' % path)
    isfile = os.path.isfile(path)
    if isfile:
        if not file_selector or file_selector(path):
            result.append(path)
    else:
        if not dir_selector or dir_selector(path):
            for sub in os.listdir(path):
                sub_path = os.path.join(path, sub)
                get_all_files(sub_path, file_selector, dir_selector, result)
    return result


if __name__ == '__main__':
    images = get_all_files('data')
    print('%d images found.' % len(images))
