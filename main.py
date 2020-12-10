import numpy as np
import cv2
import csv
from scipy.stats import kurtosis as kurt, skew

maxHrBpm = 220.0
minHrBpm = 27.0

maxHR = 1000.0/(maxHrBpm/60.0) # 220 bpm
minHR = 1000.0/(minHrBpm/60.0) # 27 bpm
hr = 0

cap = cv2.VideoCapture("Amnesia ECG_EEG.mkv")
filepath = 'Amnesia.txt'

#cap = cv2.VideoCapture("kw_2.mp4")
#filepath = 'kw_2.txt'

fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(width//2),int(height//2)))

ecgDataPoints = []
peakTimeStamps = []
hrList = []
hrListPlot = []
deltasHist = []
ecgDataPointsNpNorm = []

magKurtList = []
deltasKurtList = []
magSkewList = []
deltasSkewList = []

pointCount = 0

minRead = 99999
maxRead = 0
minDelta = 99999
maxDelta = -99999

continuousPeaks = 0
kurtError = 0
skewError = 0

with open(filepath) as fp:
    cnt = 0 # initialise counter for number of datapoints for current video frame to 0
    frameCnt = 0 # initialise counter for number of video frames to 0

    line = fp.readline() # read a single line from the dataset
    ret, frame = cap.read() # read a single frame from the video

    overlay = np.zeros((int(height//2), 350, 3), dtype=np.uint8)

    while line: # loop while end of file is not reached
        
        # check if current line is a header line and discard if so
        if line[0] is "#": 
            print(line) # ensure that file is properly loaded
            pass
        
        # current line is actual datapoint
        else:
            cnt += 1 # increment number of datapoints for current video frame read by 1
            ecgDataPoints.append(int(line.split()[5])) # obtain the fifth element in the line, casting to integer and appending to the list of ecg datapoints
            pointCount += 1 # increment number of datapoints by 1

        if pointCount > 2: # ensure more than 2 points computed
            ecgDataPointsNp = np.asarray(ecgDataPoints, dtype = np.float32) # convert to numpy array for easier operations

            # obtain minimum, maximum and range of ecg datapoints for normalisation
            minRead = min(ecgDataPointsNp) 
            maxRead = max(ecgDataPointsNp) 
            rangeRead = maxRead - minRead

            # perform normalisation and compute histogram for ecg datapoints
            ecgDataPointsNpNorm = (ecgDataPointsNp - minRead) / rangeRead
            ecgDataPointsNpNormHist = np.histogram(ecgDataPointsNpNorm, 100)

            # compute deltas from datapoints
            deltas = ecgDataPointsNpNorm[:-1] - ecgDataPointsNpNorm[1:]

            # obtain minimum, maximum and range of deltas for normalisation
            minDelta = min(deltas) 
            maxDelta = max(deltas)
            rangeDelta = maxDelta - minDelta

            # perform normalisation and compute histogram for deltas
            deltasNorm = (deltas - minDelta)/rangeDelta
            deltasHist = np.histogram(deltasNorm, 100)

            # check that current gradient is above a threshold value
            # and is either the first reading or greater than the period associated with maximum heart rate
            if deltasNorm[-1] > 0.9 and (len(peakTimeStamps) is 0 or (pointCount - peakTimeStamps[-1]) > maxHR):
                magKurtList.append(kurt(ecgDataPointsNpNorm))
                deltasKurtList.append(kurt(deltasNorm))
                kurtError = round(abs(np.mean(magKurtList) - magKurtList[-1]) + abs(np.mean(deltasKurtList) - deltasKurtList[-1]))
                #print(kurtError)   

                magSkewList.append(skew(ecgDataPointsNpNorm))
                deltasSkewList.append(skew(deltasNorm))
                skewError = round(abs(np.mean(magSkewList) - magSkewList[-1]) + abs(np.mean(deltasSkewList) - deltasSkewList[-1]))
                #print(skewError)  
               
                # append current time stamp to list
                peakTimeStamps.append(pointCount)

                continuousPeaks += 1
            
                # trim number of ecg datapoints used, 
                # ensuring that at least one period is captured by using the period associated with minimum heart rate
                # if number of data points collected is insufficient, use all available
                ecgDataPoints = ecgDataPoints[max(-int(minHR), -pointCount):]

                if continuousPeaks >= 4: # len(peakTimeStamps) >= 4: # ensure that at least 4 readings (3 intervals) have been collected for averaging
                    hr = 60.0 / (((peakTimeStamps[-1] - peakTimeStamps[-4])/3.0) / 1000.0) # compute heart rate (bpm)
                    hrList.append([pointCount, hr, kurtError]) # add timestamp and bpm for writing to csv upon completion
                    hrListPlot.append(hr)

                    print("hr: " + str(hr) + " max hr: " + str(max(hrList, key=lambda x: x[1])[1]))

                else:
                    hr = 0.0 

            # check if no peak gradient has been detected after time elapsed corresponds to min bpm
            elif (pointCount - peakTimeStamps[-1]) > minHR: 
                hr = 999 # return error code
                continuousPeaks = 0
                ecgDataPoints = ecgDataPoints[max(-int(minHR), -pointCount):] # trim datapoints to prevent erronous normalisation

        line = fp.readline() # read next datapoint

        # visualisation related code
        if cnt >= 1000.0 / fps: # sync bitalino datarate (1000hz) with video framerate
            ret, frame = cap.read() # read single video frame
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2)) # downsample video frame
            frame[:, :350, :] = cv2.addWeighted(overlay,0.4,frame[:, :350, :],0.6,0)

            frameCnt += 1 # increment frame counter
            cnt = 0 # reset counter for datapoints in current frame

            col = min(255, kurtError * 5)

            for i, val in enumerate(ecgDataPointsNpNormHist[0]):
                mag = int(100 * (val - min(ecgDataPointsNpNormHist[0]))/(max(ecgDataPointsNpNormHist[0]) - min(ecgDataPointsNpNormHist[0])))
                cv2.line(frame, (50, 250 + i), (50 + mag, 250 + i), (0,0,col))
                
            for i, val in enumerate(deltasHist[0]):
                mag = int(100 * (val - min(deltasHist[0]))/(max(deltasHist[0]) - min(deltasHist[0])))
                cv2.line(frame, (200, 250 + i), (200 + mag, 250 + i), (0,0,col))

            cv2.putText(frame, "kurt error: " + str(round(kurtError)), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)
            cv2.putText(frame, "skew error: " + str(round(skewError)), (50,380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)

            # obtain minimum, maximum and range of ecg datapoints to be plotted for autoscaling
            if len(hrListPlot):
                minHrPlot = min(hrListPlot)
                maxHrPlot = max(hrListPlot)
                hrRange = max(maxHrPlot - minHrPlot, 0.0001) # ensure range not 0 to prevent /0 error

                hrPlotPointsNp = np.asarray(hrListPlot, dtype = np.float32)
                hrPlotPointsNorm = 50.0 * (hrPlotPointsNp - minHrPlot)/hrRange

                cv2.putText(frame, "min: " + str(round(minHrPlot)) + "bpm", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)
                cv2.putText(frame, "max: " + str(round(maxHrPlot)) + "bpm", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)

                #print(hrPlotPointsNorm)

                for i, val in enumerate(hrPlotPointsNorm):
                    cv2.circle(frame, (50 + round(250.0*i/len(hrPlotPointsNorm)), 200 - int(val)), 2, (128,128,128), thickness=-1)

            plotPoints = ecgDataPoints[max(-(frame.shape[1] - 40), -pointCount):] # grab exactly enough points to fill width of frame
            plotDeltas = deltasNorm[max(-(frame.shape[1] - 40), -pointCount):] # do the same for gradients

            # obtain minimum, maximum and range of ecg datapoints to be plotted for autoscaling
            mini = min(ecgDataPoints)
            maxi = max(ecgDataPoints)
            rangi = max(maxi - mini, 0.0001) # ensure range not 0 to prevent /0 error

            plotPointsNp = np.asarray(plotPoints, dtype = np.float32)
            plotPointsNorm = (plotPointsNp - mini)/rangi

            for i, val in enumerate(plotPointsNorm):

                gradViz = int(256.0 * (plotDeltas[max(0,i-1)] - 0.5))
                cv2.circle(frame, (20 + i, int((frame.shape[0] - 20) - (val*(frame.shape[0] - 40)))), 2, (128,128,128 + gradViz), thickness=-1)

            if ret:
                intensity = int(ecgDataPointsNpNorm[-1] * 255.0) if hr != 0 else 64
                hrString = str(round(hr)) if hr <= maxHrBpm and hr >= minHrBpm else "Initialising... " if hr == 0 else "ERROR "
                cv2.putText(frame, hrString + "bpm", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (intensity,intensity,intensity), 2, cv2.LINE_AA)

                #cv2.imshow('frame', frame)
                out.write(frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

        #print((len(ecgDataPoints), len(peakTimeStamps), len(hrList)))

with open("hr.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for i in hrList:
        wr.writerow(i)
        