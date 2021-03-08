def video_to_im(path , train_path , val_path , test_path , resize_tuple):
    """
    Converts each frame to a .jpg image and save it to the defined path.
    """
    os.mkdir(common_path + "/" + train_path)
    os.mkdir(common_path + "/" + val_path)
    os.mkdir(common_path + "/" + test_path)

    frame = 0
    cap = cv2.VideoCapture(path)

    while True:
        check , img = cap.read()
        if(check):
            if frame < 3600:
                path =  common_path + "/" + train_path

            elif frame >= 3600 and frame < 5000:
                path = common_path + "/" + val_path

            else:
                path = common_path + "/" + test_path

            img = cv2.resize(img, resize_tuple)
            cv2.imwrite(os.path.join(path, str(frame) + ".jpg"), img)

            frame += 1
            print("Processed frame :{}".format(frame))
        else:
            break

    cap.release()

video_to_im(video_file , "train" , "val" , "test" , (1920 // 2  , 1080 // 2))
