# refer: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import time
import sys
import os, urllib, cv2

# 主动添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))))


# 预览摄像头数据
# def view_camera():
#     pass

def get_image_name():
    """
    保存图片 ，在目标文件夹超过指定数量时按时间顺序覆盖
    :param frame: 传入待写入图片
    :param save_struct_time: 图片保存结构化时间
    :return: None 写入图片
    """
    out_image_folder = 'module/camera/hand_keypoints/images_out/'
    sava_path = os.path.join(
        out_image_folder, time.strftime(
            "%Y%m%d%H%M%S", time.localtime()) + '.jpg')
    # print(sava_path)
    save_image_len = len(os.listdir(out_image_folder))
    print(os.listdir(out_image_folder))
    images_list = sorted([int(i.split('.')[0])
                          for i in os.listdir(out_image_folder) if i.endswith('jpg')])
    last_image_name_0 = os.path.join(out_image_folder, str(images_list[-1]) + '.jpg')
    last_image_name_1 = os.path.join(out_image_folder, str(images_list[-1]) + '.jpg')
    return [last_image_name_0,last_image_name_1]


def car_control():
    choose_move_status = st.radio('主动选择运动', ('停止', '前进', '后退', '左转', '右转'))
    st.write(choose_move_status)
    #  TODO 显示视频  根据手势自动运行
    st.image(get_image_name())


def arm_control():
    choose_move_status = st.multiselect('选择Dof', ('1轴', '2轴', '3轴', '4轴', '5轴', '6轴'))
    angel = st.slider('选择舵机旋转角度', 0, 180, 90)
    st.write(angel)
    #  TODO 显示视频
    st.image(get_image_name())


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown("instructions.md")

    # Download external dependencies.
    # for filename in EXTERNAL_DEPENDENCIES.keys():
    #     download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("选择功能")
    app_mode = st.sidebar.selectbox("功能",
                                    ["说明", "预览摄像头", "car控制", "arm控制"])
    if app_mode == "说明":
        st.sidebar.success('各功能说明')

    elif app_mode == "预览摄像头":
        st.sidebar.success('To continue select "Run the app".')

    elif app_mode == "car控制":
        readme_text.empty()
        car_control()

    elif app_mode == "arm控制":
        readme_text.empty()
        arm_control()


if __name__ == "__main__":
    main()
