import cv2
import numpy as np
import glob
import os


def get_all_image_input(directory):
    list_image = glob.glob(directory + "*.jpg")
    list_image.extend(glob.glob(directory + "*.png"))
    return list_image


def main(list_input):
    for image in list_input:
        # Tạo thư mục output
        basename = os.path.basename(image)
        filename = os.path.splitext(basename)[0]
        dir = './output/' + filename
        if not os.path.exists(dir):
            os.mkdir(dir)

        # Import ảnh
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # Lấy thông số dài rộng của ảnh
        heigh, width = img.shape[:2]
        if heigh > 1000 and width > 1000:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        # img=cv2.GaussianBlur(img,)

        # Chọn ngưỡng ảnh
        (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        cv2.imwrite("./tmp/" + filename + "_threshold.png", img_bin)
        # Invert
        img_bin = 255 - img_bin

        cv2.imwrite("./tmp/" + filename + "_inverted.png", img_bin)
        # exit()

        # Lọc các đường kẻ ngang trong ảnh
        kernel_length = np.array(img).shape[1] // 80
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

        # Tìm các contour của ảnh
        im2, contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        list_mark = []
        # Đảo ngược thứ tự list các contours
        contours.reverse()
        print(filename + "--------------------")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(x, y, w, h)
            if x <= 20:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

                if len(list_mark) % 2 == 0:
                    list_mark.append(y + h)
                    # cv2.line(img, (0, y + h), (width, y + h), (0, 255, 0), 1)
                else:
                    list_mark.append(y)
                    # cv2.line(img, (0, y), (width, y), (0, 255, 0), 1)

        if len(list_mark) % 2 != 0:
            print("Wrong detection")
        else:
            index = 0
            for i in range(0, len(list_mark), 2):
                de_bai = img[list_mark[i]:list_mark[i + 1], 0:width]
                cv2.imwrite('./output/' + filename + '/' + str(index) + '.png', de_bai)
                index += 1

        cv2.imwrite("./tmp/" + filename + "_img_horizon.png", img)


def show_image(img):
    cv2.imshow("image" + str(img.shape[0]), img)
    cv2.waitKey(0)


"""
Cắt ảnh dựa trên các nét đánh dấu
Input: list ảnh trong thư mục input/
Output: list ảnh trong thư mục output tương ứng với từng ảnh
"""
if __name__ == "__main__":
    input = get_all_image_input("./input/")
    main(input)
