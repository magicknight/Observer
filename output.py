__author__ = 'zhihua'


def output(file_name, position, confidence, output_file):
    """

    :param file_name:
    :param position:
    :param confidence:
    :param output_file:
    """
    lesion = 1 if int(file_name.split('lesion_')[-1].split('_')[0]) > 0 else 0  # if or not this image contains lesion
    truth_x = file_name.split('x_')[-1].split('_')[0]
    truth_y = file_name.split('z_')[-1].split('_')[0]
    output_file.write(file_name + ' ' + file_name + ' ' + str(confidence) + ' ' + str(lesion) + ' 1 '
                      + str(position[0]) + ' ' + str(position[1]) + ' ' + truth_x + ' ' + truth_y + '\n')
