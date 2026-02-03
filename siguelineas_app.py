list_ListaLinea = RmList()
pid_PID = PIDCtrl()
variable_x = 0
variable_v_average = 0
variable_k = 0
variable_v = 0
def start():
    global variable_x
    global variable_v_average
    global variable_k
    global variable_v
    global list_ListaLinea
    global pid_PID
    robot_ctrl.set_mode(rm_define.robot_mode_chassis_follow)
    gimbal_ctrl.rotate_with_degree(rm_define.gimbal_down,55)
    vision_ctrl.enable_detection(rm_define.vision_detection_line)
    vision_ctrl.line_follow_color_set(rm_define.line_follow_color_blue)
    pid_PID.set_ctrl_params(200,0.01,5)
    while True:
        list_ListaLinea=RmList(vision_ctrl.get_line_detection_info())
        if len(list_ListaLinea) == 42:
            if list_ListaLinea[2] == 1:
                variable_x = list_ListaLinea[19]
                pid_PID.set_error(variable_x - 0.5)
                gimbal_ctrl.rotate_with_speed(pid_PID.get_output(),0)
                chassis_ctrl.set_trans_speed(0.5)
                chassis_ctrl.move(0)
        else:
            gimbal_ctrl.rotate_with_speed(50,0)
