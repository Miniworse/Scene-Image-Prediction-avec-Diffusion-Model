import matlab.engine
import numpy as np

try:
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()

    root_path = r'D:\Documents\Self_Files\ForwardBackward Process\projects\正反演代码整理版_96 - 已修正\正反演代码整理版_96 - 已修正'
    eng.addpath(eng.genpath(root_path), nargout=0)
    eng.cd(root_path, nargout=0)

    # eng.run('WorkSpace\\demo.m' ,nargout=0)
    Param = eng.Param4python()

    TB = np.zeros([256,256], dtype=np.double)
    TB[100:150, 100:150] = 1

    res, R = eng.TB4python(Param, TB, nargout=2)

    res = np.array(res, dtype=np.complex128)
    R = np.array(R, dtype=np.complex128)

    # 关闭 MATLAB 引擎
    eng.quit()

except Exception as e:
    print("调用 MATLAB 出错:", e)

# print(test)
