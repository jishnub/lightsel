from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pyfits
import scipy.stats
import os


def get_cleaned_data(fitsfile):
    #~ The one where they bin the data
    #######################################################################################################
    #~ Load fits file and read data
    #~ Try reading npz file if possible as it's faster than reading fits
    #~ Create the file if not there so that reading will be faster second time onwards
    #~ fitsfile = 'data_1.fits'
    print "Reading file",fitsfile
    filename = os.path.splitext(os.path.split(fitsfile)[-1])[0]
    directory = os.path.dirname(fitsfile)
    npzfile = os.path.splitext(fitsfile)[0]+'.npz'
    if os.path.exists(npzfile):
        datafile=np.load(npzfile)
        times = datafile['times']
        counts = datafile['counts']
    else:
        data = np.array(pyfits.getdata(fitsfile))
        times,counts = map(np.array,zip(*data))
        np.savez(npzfile,times=times,counts=counts)

    times -= times[0]

    #~ Get limits for binning
    largest_integer_time = int(times[-1])
    smallest_integer_time = int(times[0])

    #~ Create bins from smallest_integer_time to largest_integer_time as edges. Number of bins would be 
    #~ largest_integer_time - smallest_integer_time - 1
    counts_1sec_bin,time_1sec_bin_edges,_ = scipy.stats.binned_statistic(times,counts,statistic='sum',
    bins=largest_integer_time - smallest_integer_time,range=(smallest_integer_time,largest_integer_time))

    #~ Get bin centers from bin edges, these will be shifted by 0.5 sec wrt edges
    time_1sec_bin_centers = time_1sec_bin_edges[:-1]+0.5


    #~ The one where they add lines to the plot
    ##########################################################################################################
    start_edges = []
    stop_edges = []
    zoom_spans = []

    button_pressed = {"shift":False,"control":False}

    #~ Define what happens when you click on a line or on a span
    #~ This will delete lines and spans
    def onpick(event):
        
        artist = event.artist
        mouseevent = event.mouseevent
        
        if type(artist)==matplotlib.lines.Line2D:

            #~ Find the object that matches the line and remove it
            #~ This clears the list, but not the plot
            
            for lineobj in start_edges:
                if lineobj.line() == artist:
                    for span in zoom_spans:
                        if span.start_line == artist:
                            span.remove()
                            zoom_spans.remove(span)
                    start_edges.remove(lineobj)
                    
            for lineobj in stop_edges:
                if lineobj.line() == artist:
                    for span in zoom_spans:
                        if span.stop_line == artist:
                            span.remove()
                            zoom_spans.remove(span)
                    stop_edges.remove(lineobj)
            
            #~ Remove the line from the plot
            artist.remove()
            
        elif type(artist) == matplotlib.patches.Polygon:
            
            #~ Find the object that matches the span and remove it
            #~ This clears the list, but not the plot
            
            for spanobj in zoom_spans:
                if spanobj.span() == artist:
                    fig = spanobj.figure()
                    plt.close(fig)
                    zoom_spans.remove(spanobj)
            
            #~ Remove the polygon from plot
            artist.remove()
            
        plt.draw()
            
    #~ Define what happens on clicking the plot
    #~ This will add lines and spans to the plot
    def onclick(event):
        #~ Note down start time if shift+left click
        if event.button == 1 and button_pressed["shift"]:

            start_edges.append(Edge(event.xdata,"green"))
            
        #~ Note down end time if shift+right click
        elif event.button == 3 and button_pressed["shift"]:
            
            stop_edges.append(Edge(event.xdata,"red"))

        elif event.button == 1 and button_pressed["control"]:
            eventx = event.xdata
            
            #~ start_times = get_times_from_boundaryobj_list(start_edges)
            #~ stop_times = get_times_from_boundaryobj_list(stop_edges)
            
            start_edges_below = filter(lambda x: x.timecoord()<eventx,start_edges)
            if not start_edges_below: return
            else: lower_start = max(start_edges_below,key=lambda x: x.timecoord())
            
            stop_edges_above = filter(lambda x: x.timecoord()>eventx,stop_edges)
            if not stop_edges_above: return
            else: upper_stop = min(stop_edges_above,key=lambda x: x.timecoord())
            
            start_edges_above = filter(lambda x: x.timecoord()>eventx,start_edges)
            if start_edges_above: upper_start = min(start_edges_above,key=lambda x: x.timecoord())
            else: upper_start = None
            
            stop_edges_below = filter(lambda x: x.timecoord()<eventx,stop_edges)
            if stop_edges_below: lower_stop = max(stop_edges_below,key=lambda x: x.timecoord())
            else: lower_stop=None
            
            #~ print "lower_start",lower_start
            #~ print "upper_stop",upper_stop
            #~ print "lower_stop",lower_stop
            #~ print "upper_start",upper_start
            
            if upper_start is not None and upper_start.timecoord()<upper_stop.timecoord(): return
            if lower_stop is not None and lower_stop.timecoord() > lower_start.timecoord() : return

            spanobj = Span(lower_start,upper_stop)
            zoom_spans.append(spanobj)
            spanobj.plot()

        plt.draw()

    #~ Check if shift is pressed. Note down times only if shift is pressed
    def onpress(event):
        if event.key=="shift":
            button_pressed["shift"] = True
        elif event.key=="control":
            button_pressed["control"] = True
        elif event.key == "d":
            
            for line in get_lines_from_boundaryobj(start_edges+stop_edges):
                line.remove()
            
            while start_edges: start_edges.pop()
            while stop_edges: stop_edges.pop()
            
            plt.draw()
            
    def onrelease(event):
        if event.key=="shift":
            button_pressed["shift"] = False
        elif event.key=="control":
            button_pressed["control"] = False
            
            
    #~ The one where they create the line and span objects
    ##############################################################################################################

    class Edge():
        def __init__(self,xdata,color):
            self.time = xdata
            self.vline = plt.axvline(self.time,color=color,linewidth=2,picker=True)
            
        def timecoord(self):
            return self.time
            
        def line(self):
            return self.vline

    def get_times_from_boundaryobj_list(list_of_objects):
        return map(lambda x: x.timecoord(),list_of_objects)

    def get_lines_from_boundaryobj(list_of_objects):
        return map(lambda x: x.line(),list_of_objects)

    class Span():
        def __init__(self,leftline,rightline):
            self.start_time = leftline.timecoord()
            self.stop_time = rightline.timecoord()
            
            self.start_line = leftline.line()
            self.stop_line = rightline.line()
            
            self.axvspan=None
            self.zoomfig=None
            
            self.ind = np.where((times>self.start_time) & (times<=self.stop_time))[0]
            self.ind = self.ind[:-(len(self.ind)%1024)]
        
            if not self.ind.any():
                print "No points in range, increase range width to include at least 1024 points"
            
        def plot(self):
            plt.figure(0)
            self.axvspan = plt.axvspan(self.start_time,self.stop_time,color='skyblue',picker=True)
            plt.draw()
            sel_times = times[self.ind]
            sel_counts = counts[self.ind]
            self.zoomfig = plt.figure()
            plt.errorbar(sel_times,sel_counts,yerr=np.sqrt(sel_counts))
            ax=plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(5,prune='lower'))
            ax.yaxis.set_major_locator(MaxNLocator(5,prune='lower'))
            ax.xaxis.major.formatter._useMathText = True
            plt.tick_params(axis='both',labelsize=14)
            plt.xlabel("Time (sec)",fontsize=20)
            plt.ylabel("Counts",fontsize=20)
            plt.title("64 ms time bin",fontsize=20)
            plt.tight_layout()
            plt.connect('key_press_event', onpress)
            plt.connect('key_release_event', onrelease)
            plt.show(block=False)
                
        def span(self):
            return self.axvspan
            
        def figure(self): 
            return self.zoomfig
        
        def remove(self):
            self.axvspan.remove()
            plt.close(self.zoomfig)
            plt.draw()

    #~ The one where they list the start and stop times
    ###########################################################################################################

    plt.figure(0)

    plt.errorbar(time_1sec_bin_centers,counts_1sec_bin,yerr=np.sqrt(counts_1sec_bin))

    plt.connect('button_press_event', onclick)
    plt.connect('key_press_event', onpress)
    plt.connect('key_release_event', onrelease)
    plt.connect('pick_event', onpick)


    plt.xlabel("Time (sec)",fontsize=20)
    plt.ylabel("Counts",fontsize=20)
    plt.title("1 sec time bin",fontsize=20)
    plt.tick_params(axis="both",which="major",labelsize=14)
    ax = plt.gca()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax.xaxis.major.formatter._useMathText = True
    plt.tight_layout()

    plt.show()

    start_times = sorted(get_times_from_boundaryobj_list(start_edges))
    stop_times = sorted(get_times_from_boundaryobj_list(stop_edges))

    assert len(start_times)==len(stop_times),"Number of start times is different from number of stop times"

    print "start_times=",start_times
    print "stop_times=",stop_times
    print "intervals=",zip(start_times,stop_times)

    start_times = np.array(start_times)
    stop_times = np.array(stop_times)

    if (not start_times.any()) or (not stop_times.any()):
        print "no selection, quitting"
        quit()



    #~ The one where they write out the 64ms counts in the (start,stop] interval
    ###############################################################################################################


    sel_time = []
    sel_counts = []
    sel_c_err = []

    for x in range(np.size(start_times)):
        ind = (np.where((times>start_times[x]) & (times<=stop_times[x]))[0])
        ind = ind[:-(len(ind)%1024)]
        if not ind.any(): continue
        if x==0:
            sel_time = times[ind]
            sel_counts =counts[ind]
        else:
            sel_time = np.concatenate((sel_time,times[ind]),    axis=0)
            sel_counts = np.concatenate((sel_counts, counts[ind]),  axis=0)
            
    sel_counts = np.array(sel_counts)
    sel_time = np.array(sel_time)

    file_to_save_to = os.path.join(directory,filename+"_cleaned.qdp")
    print "saving to",file_to_save_to
    np.savetxt(file_to_save_to, np.transpose([sel_time, sel_counts]), fmt=["%.6f","%.6f"])

#~ get_cleaned_data('data_1.fits')
