################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#                
#                Centre for Speech Technology Research                 
#                     University of Edinburgh, UK                       
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.                           
#                                                                       
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute  
#  this software and its documentation without restriction, including   
#  without limitation the rights to use, copy, modify, merge, publish,  
#  distribute, sublicense, and/or sell copies of this work, and to      
#  permit persons to whom this work is furnished to do so, subject to   
#  the following conditions:
#  
#   - Redistributions of source code must retain the above copyright  
#     notice, this list of conditions and the following disclaimer.   
#   - Redistributions in binary form must reproduce the above         
#     copyright notice, this list of conditions and the following     
#     disclaimer in the documentation and/or other materials provided 
#     with the distribution.                                          
#   - The authors' names may not be used to endorse or promote products derived 
#     from this software without specific prior written permission.   
#                                  
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       
#  THIS SOFTWARE.
################################################################################

# NOTES
# still to consider: pygal, for HTML5 SVG plotting

import math
import string
import os

# this module provides the base classes that we specialise here
import logging # as logging

# for plotting
import matplotlib

# should make this user-configurable - TO DO later
# this line has to come before the import of matplotlib.pyplot
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import pylab

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# matplotlib needs to be passed numpy arrays
import numpy

# for sorting tuples
from operator import itemgetter, attrgetter


# TO DO - this needs to be attached to the logging module so that it's available via config options
# class PlotHandler(logging.FileHandler):
#     """A handler for saving plots to disk"""
#     def __init__(self,filename):
#         logging.FileHandler.__init__(self,filename, mode='a', encoding=None, delay=False)
    


class PlotWithData(object):
    # a generic plot object that contains both the underlying data and the plot itself
    # this class needs to be subclassed for each specialised type of plot that we want
    
    # the underlying data for the plot - a dictionary of data series
    # each series is a list of data points of arbitrary type (e.g., tuples, arrays, ..)
    data=None
    # the plot generated from these data
    plot=None

    def __init__(self,name):
        # clear the data series
        self.data={}
        
    def add_data_point(self,series_name,data_point):
        # if there is no data series with this name yet, create an empty one
        if not self.data.has_key(series_name):
            self.data[series_name]=[]
        # append this data point (e.g., it might be a tuple (x,y) )
        # don't worry about data type or sorting - that is not our concern here
        self.data[series_name].append(data_point)

    def sort_and_validate(self):
        # only applied if the data points are tuples, such as (x,y) values
        
        # TO DO: first check that each series is a list of tuples, and that they have the same number of elements

        # this method checks that all data series
        # 1. have the same length
        # 2. are sorted in ascending order of x
        # 3. have identical values in their x series


        # there has to be at least one data series
        try:
            assert len(self.data) > 0
        except AssertionError:
            logger.critical('No data series found in plot')
            raise

        # check lengths are consistent, sort, then check x values are identical
        l=-1
        reference_x=None
        # print "starting with self.data=",self.data
        for series_name,data_points in self.data.iteritems():
            if l > 0:
                assert l == len(data_points)
            else:
                l = len(data_points)
            # sort by ascending x value
            data_points.sort(key=itemgetter(0))

            if reference_x:
                assert reference_x == [seq[0] for seq in data_points]
            else:
                # extract a list of just the x values
                reference_x = [seq[0] for seq in data_points]


        # print "ending with self.data=",self.data
    
    def generate_plot(self,**kwargs):
        logger = logging.getLogger("plotting")
        logger.error('Cannot generate a plot from abstract class: PlotWithData' )
        # raise an exception here?

class MultipleSeriesPlot(PlotWithData):

    def generate_plot(self,filename,title='',xlabel='',ylabel='',xlim=None,ylim=None):
        
        logger = logging.getLogger("plotting")
        logger.debug('MultipleSeriesPlot.generate_plot')
        
        # a plot with one or more time series sharing a common x axis:
        # e.g., the training error and the validation error plotted against epochs

        # sort the data series and make sure they are consistent
        self.sort_and_validate()

        # if there is a plot already in existence, we will clear it and re-use it;
        # this avoids creating extraneous figures which will stay in memory
        # (even if we are no longer referencing them)
        if self.plot:
            self.plot.clf()
        else:
            # create a plot
            self.plot = plt.figure()

        splt = self.plot.add_subplot(1, 1, 1)
        splt.set_title(title)
        splt.set_xlabel(xlabel)
        splt.set_ylabel(ylabel)
        
        if xlim:
            pylab.xlim(xlim)
        if ylim:
            pylab.ylim(ylim)
        
        for series_name,data_points in self.data.iteritems():
            xpoints=numpy.asarray([seq[0] for seq in data_points])
            ypoints=numpy.asarray([seq[1] for seq in data_points])
            line, = splt.plot(xpoints, ypoints, '-', linewidth=2)
            logger.debug('set_label for %s' % series_name)
            line.set_label(series_name)

        splt.legend()
        
        # TO DO - better filename configuration for plots
        self.plot.savefig(filename)

class SingleWeightMatrixPlot(PlotWithData):

    def generate_plot(self, filename, title='', xlabel='', ylabel=''):

        data_keys = self.data.keys()
        key_num = len(data_keys)

        self.plot = plt.figure()
        if key_num == 1:   
            splt = self.plot.add_subplot(1, 1, 1)
            im_data = splt.imshow(numpy.flipud(self.data[data_keys[0]][0]), origin='lower')
            splt.set_xlabel(xlabel)
            splt.set_ylabel(ylabel)
            splt.set_title(title)
        else:   ## still plotting multiple image in one figure still has problem. the visualization is not good
            logger.error('no supported yet')

        self.plot.colorbar(im_data)
        self.plot.savefig(filename)  #, bbox_inches='tight'

#class MultipleLinesPlot(PlotWithData):
#    def generate_plot(self, filename, title='', xlabel='', ylabel=''):    

class LoggerPlotter(logging.getLoggerClass()):
    """Based on the built-in logging class, with added capabilities including plotting"""
    
    # a dictionary to store all generated plots
    # keys are plot names
    # values are 
    plots ={}
    # where the plots will be saved - a directory
    plot_path='/tmp' # default location
        
    def __init__(self,name):
        # initialise the logging parent class
        # (should really use 'super' here I think, but that fails - perhaps because the built in logger class is not derived from 'object' ?)
        logging.Logger.__init__(self,name)

    def set_plot_path(self,path):
        self.plot_path = path

    def remove_all_plots(self):
        self.plots={}

    def create_plot(self,plot_name,plot_object):
        self.plots[plot_name] = plot_object(plot_name)
        
    def add_plot_point(self,plot_name,series_name,data_point):
        # add a data point to a named plot
        if not self.plots.has_key(plot_name):
            self.plots[plot_name] = PlotWithData(plot_name)
        self.plots[plot_name].add_data_point(series_name,data_point)
            
    def save_plot(self,plot_name,**kwargs):
        logger = logging.getLogger("plotting")
        if not self.plots.has_key(plot_name):
            logger.warn('Tried to generate a plot called %s that does not exist' % plot_name)
            # raise an exception here?
        else:
            # # the filename to save to is known by the handler, which needs to be assigned to this logger
            # # look at the handlers attached to this logger instance
            # ph=None
            # for h in self.handlers:
            #     # we want an instance of a PlotHandler - we'll take the first one we find
            #     # (behaviour will be unpredictable if there is more than one handler of this type)
            #     if isinstance(h,PlotHandler):
            #         ph=h
            #         break
            # if ph:
            # TO DO - need to be sure of safe file names
            if not os.path.isdir(self.plot_path):
                os.makedirs(self.plot_path)
            filename = self.plot_path + "/" + string.replace(plot_name, " ", "_") + ".pdf"
            logger.info('Generating a plot in file %s' % filename)
            self.plots[plot_name].generate_plot(filename,**kwargs)
            # else:
            #     logger.warn('No handler of type PlotHandler is attached to this logger - cannot save plots')




class ColouredFormatter(logging.Formatter):

    # colourising formatter adapted from an answer to this question on Stack Overflow
    # http://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    COLOURS = {
        'DEBUG': BLUE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': MAGENTA
    }

    max_level_name_width = '8' 

    # terminal escape sequences
    RESET_SEQ = "\033[0m"
    COLOUR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    def __init__(self, fmt=None, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        if record.levelname in self.COLOURS:
            # pad to fixed width - currently hardwired, should make this dynamic
            # maximum width of level names, which is the 8 characters of "CRITICAL"
            fixed_width_levelname = '{0:8s}'.format(record.levelname)
            record.name = '{0:8s}'.format(record.name)
            # The background is set with 40 plus the number of the color, and the foreground with 30
            record.levelname = self.COLOUR_SEQ % (30 + self.COLOURS[record.levelname]) + fixed_width_levelname + self.RESET_SEQ
        return logging.Formatter.format(self, record)

    def factory(fmt, datefmt):
        default = logging.Formatter(fmt, datefmt)
        return ColouredFormatter(default)

if __name__ == '__main__':
    # some simple tests

    # tell the built-in logger module to use our custom class when instantiating any new logger
    logging.setLoggerClass(LoggerPlotter)


    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)

    # a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = ColouredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)



    print "testing the logging code"
    logger.debug('A DEBUG message')
    logger.info('A INFO message')
    logger.warning('A WARN message')
    logger.error('A ERROR message')
    logger.critical('A CRITICAL message')


    plotlogger = logging.getLogger("plotting")
    plotlogger.setLevel(logging.DEBUG)
    # handler for plotting logger - will write only to console
    plotlogger.addHandler(ch)


    # # need a handler which will control where to save plots
    # ph = PlotHandler("/tmp/plot_test/testing.pdf")
    # logger.addHandler(ph)
    

    print "testing the plotting code"
    
    # the first argument is just a key for referring to this plot within the code
    # the second argument says what kind of plot we will be making


    plotlogger.set_plot_path("./tmp")
    
    logger.create_plot('test plot',MultipleTimeSeriesPlot)
    
    plotlogger.add_plot_point('test plot','validation',(1,4))
    plotlogger.add_plot_point('test plot','validation',(3,2))
    plotlogger.add_plot_point('test plot','validation',(2,3))
    plotlogger.add_plot_point('test plot','validation',(4,3))
    
    plotlogger.add_plot_point('test plot','training',(1,3))
    plotlogger.add_plot_point('test plot','training',(3,1))
    plotlogger.add_plot_point('test plot','training',(2,2))
    plotlogger.add_plot_point('test plot','training',(4,4))
    
    plotlogger.save_plot('test plot',title='Training and validation error',xlabel='epochs',ylabel='error')

    weights = [[1, 2, 3, 3], [1, 1, 2, 1], [2, 1, 2, 2]]
    logger.create_plot('activation weight', SingleWeightMatrixPlot)
    plotlogger.add_plot_point('activation weight', 'weight1', weights)
    plotlogger.add_plot_point('activation weight', 'weight2', weights)
    plotlogger.add_plot_point('activation weight', 'weight3', weights)

    plotlogger.save_plot('activation weight', title='weight', xlabel='dimension', ylabel='dimension')
