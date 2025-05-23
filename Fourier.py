# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:58:00 2022

@author: oscar
"""
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.widgets as widgets
fc=10       #starting frequency
n=100       #data points
a=1         #wave amplitude
ti,tf=0,1   #start and end point
SR=n/(tf-ti)    #sample rate
Nq=SR/2     #Nyquist Frequency
t=np.linspace(ti,tf,n)
noise=np.random.uniform(-a,a,n) #noise with max value=amplitude
NO=0 #noise coefficient
#functions to shorten inbuilt functions
def four(x):
    return np.fft.fft(x)
def ifour(x):
    return np.fft.ifft(x)
def frec(nr,sr):
    return np.fft.fftfreq(nr,sr)
def sine(ts,f):
    return np.sin(2*np.pi*f*ts)
def square(ts,f):
    return sp.signal.square(2*np.pi*f*ts)
def saw(ts,f):
    return sp.signal.sawtooth(2*np.pi*f*ts)


F=np.abs(frec(n,tf/n)) #frequency space
i=0
#wave function
fun=a*np.array([sine(t,fc),square(t,fc),saw(t,fc)])-NO*noise
#fourier transformed wave
fun2=np.abs(four(fun))
#frequency slider function
def slideFrec(f):
    global fc
    fc=f
    global fun
    global fun2
    fun=a*np.array([sine(t,fc),square(t,fc),saw(t,fc)])-NO*noise
    fun2=np.abs(four(fun))
    axesHandle.set_ydata(fun[i])
    axesHandle2.set_ydata(fun2[i])
    plt.draw()
#change wave function
def Wavetype(label):
    global t
    global fc
    hzdict = {'sine': 0, 'square': 1, 'sawtooth': 2}
    global i
    i= hzdict[label]
    axesHandle.set_ydata(fun[i])
    axesHandle2.set_ydata(fun2[i])
    plt.draw()
#close window
def closeCallback(event):
    plt.close('all') # Close all open figure windows
#data point function
def sliderNumber(num):
    global n
    n=int(num)
    global t
    t=np.linspace(ti,tf,n)
    global F
    F=np.abs(frec(n,tf/n))
    global fun
    global fun2
    global noise
    noise=np.random.uniform(-a,a,n)
    fun=a*np.array([sine(t,fc),square(t,fc),saw(t,fc)])-NO*noise
    fun2=np.abs(four(fun))
    axesHandle.set_xdata(t)
    axesHandle2.set_xdata(F)
    axesHandle.set_ydata(fun[i])
    axesHandle2.set_ydata(fun2[i])
    plt.draw() # Redraw the axes
#amplitude function
def slideAmp(am):
    global a
    global noise
    noise=np.random.uniform(-a,a,n)
    a=float(am)
    fun=a*np.array([sine(t,fc),square(t,fc),saw(t,fc)])-NO*noise
    fun2=np.abs(four(fun))
    axesHandle.set_ydata(fun[i])
    axesHandle2.set_ydata(fun2[i])
    plt.draw()
#noise function
def slideNois(no):
    global fun
    global fun2
    global NO
    NO=no
    fun=a*np.array([sine(t,fc),square(t,fc),saw(t,fc)])-NO*noise
    fun2=np.abs(four(fun))
    axesHandle.set_ydata(fun[i])
    axesHandle2.set_ydata(fun2[i])
    plt.draw()

fig,axs=plt.subplots(1,2,figsize=(14,6))
#plot wave
axesHandle, = axs[0].plot(t, fun[i], lw=2, color='red')
#plot fourier transform
axesHandle2, = axs[1].plot(F, fun2[i], lw=2, color='blue')
A=axs.flat
A[0].set(xlabel='time [s]',ylabel='amplitude')
A[1].set(xlabel='frequency [Hz]',ylabel='amplitude^2')




# Add axes to contain the slider
sax = plt.axes([0.08, 0, 0.55, 0.03])

# Add the slider
sliderHandle = widgets.Slider(sax, 'Frequency', 0.15, 50, valinit=fc)
sliderHandle.on_changed(slideFrec)


sn = plt.axes([0.08, 0.05, 0.55, 0.03])
N=np.round(np.linspace(10,1010,1000),0)

sHn = widgets.Slider(sn, 'Number', 10, 1000,valinit=n,valstep=N)
sHn.on_changed(sliderNumber)


# Add radio buttons to change wave type
rax = plt.axes([0.9, 0.4, 0.2, 0.2])
radioHandle = widgets.RadioButtons(rax, ('sine', 'square', 'sawtooth'))
radioHandle.on_clicked(Wavetype)

samp = plt.axes([0.08, 0.95, 0.55, 0.03])

# Add the slider.
# Set slider values from 0.1 to 5.0, and set initial value of slider to 1.0
sAmp = widgets.Slider(samp, 'Amplitude', 0.00001, 1, valinit=a)
sAmp.on_changed(slideAmp)

sno = plt.axes([0.08, 0.9, 0.55, 0.03])

# Add the slider.
# Set slider values from 0.1 to 5.0, and set initial value of slider to 1.0
sNois = widgets.Slider(sno, 'Noise', 0, 1, valinit=NO)
sNois.on_changed(slideNois)

# Add button to close GUI
bax = plt.axes([0.9, 0, 0.1, 0.1]) # Add new axes to the figure
buttonHandle = widgets.Button(bax, 'Close')
buttonHandle.on_clicked(closeCallback)

