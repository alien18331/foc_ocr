import time
from collections import namedtuple

import gi
import re
import numpy
from enum import Enum
gi.require_version("Gst", "1.0")
gi.require_version("Tcam", "1.0")

from gi.repository import GLib, GObject, Gst, Tcam



DeviceInfo = namedtuple("DeviceInfo", "status name identifier connection_type")
CameraProperty = namedtuple("CameraProperty", "status value min max default step type flags category group")

class SinkFormats(Enum):
    GRAY8 = 0
    GRAY16_LE = 1
    BGRA = 2

    def toString(pf):
        if( pf == SinkFormats.GRAY16_LE ):
            return "GRAY16_LE"

        if( pf == SinkFormats.GRAY8 ):
            return "GRAY8"

        if( pf == SinkFormats.BGRA ):
            return "BGRx"

        return "BGRx"

    def fromString(pf):
        if( pf == "GRAY16_LE"):
            return SinkFormats.GRAY16_LE
        if( pf == "GRAY8"):
            return SinkFormats.GRAY8
      
        return SinkFormats.BGRA



class TIS:
    'The Imaging Source Camera'

    def __init__(self):
        ''' Constructor
        :return: none
        '''
        Gst.init([]) # Usually better to call in the main function.
        # Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
        self.serialnumber = ""
        self.height = 0
        self.width = 0
        self.framerate="15/1"
        self.livedisplay = False #True
        self.sinkformat = SinkFormats.BGRA
        self.sample = None
        self.samplelocked = False
        self.newsample = False
        self.img_mat = None
        self.ImageCallback = None
        self.pipeline = None
        self.source = None


    def openDevice(self,serial, width, height, framerate, sinkformat: SinkFormats, showvideo: bool):
        ''' Inialize a device, e.g. camera.
        :param serial: Serial number of the camera to be used.
        :param width: Width of the wanted video format
        :param height: Height of the wanted video format
        :param framerate: Numerator of the frame rate. /1 is added automatically
        :param color: True = 8 bit color, False = 8 bit mono. ToDo: Y16
        :return: none
        '''
        self.serialnumber = serial
        self.height = height
        self.width = width
        self.framerate = framerate
        self.sinkformat = sinkformat
        self.livedisplay = showvideo
        self._createPipeline()
        self.source.set_property("serial", self.serialnumber)
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.get_state(40000000)


    def _createPipeline(self):
        p = 'tcambin name=source ! capsfilter name=caps'
        if self.livedisplay is True:
            p += " ! tee name=t"
            p += " t. ! queue ! videoconvert ! ximagesink"
            p += " t. ! queue ! appsink name=sink"
        else:
            p += ' ! appsink name=sink'

        print(p)
        try:
            self.pipeline = Gst.parse_launch(p)
        except GLib.Error as error:
            print("Error creating pipeline: {0}".format(error))
            raise

        # Quere the source module.
        self.source = self.pipeline.get_by_name("source")

        # Query a pointer to the appsink, so we can assign the callback function.
        appsink = self.pipeline.get_by_name("sink")
        appsink.set_property("max-buffers",5)
        appsink.set_property("drop",1)
        appsink.set_property("emit-signals",1)
        appsink.connect('new-sample', self.on_new_buffer)


    def on_new_buffer(self, appsink):
        self.newsample = True
        if self.samplelocked is False:
            try:
                self.sample = appsink.get_property('last-sample')
                if self.ImageCallback is not None:
                    self.__convert_sample_to_numpy()
                    self.ImageCallback(self, *self.ImageCallbackData);

            except GLib.Error as error:
                print("Error on_new_buffer pipeline: {0}".format(error))
                raise
        return False


    def setSinkFormat(self, sf: SinkFormats):
        self.sinkformat = sf


    def showLive(self, show: bool):
        self.livedisplay = show


    def _setcaps(self):
        """ 
        Set pixel and sink format and frame rate
        """
        caps = Gst.Caps.new_empty()
        videoformat = 'video/x-raw,format=%s,width=%d,height=%d,framerate=%s' % ( SinkFormats.toString(self.sinkformat),self.width,self.height,self.framerate,)
        structure = Gst.Structure.new_from_string(videoformat)

        caps.append_structure(structure)
        
        capsfilter = self.pipeline.get_by_name("caps")
        capsfilter.set_property("caps", caps)


    def Start_pipeline(self):
        """
        Start the pipeline, so the video runs
        """
        try:
            self._setcaps()
            self.pipeline.set_state(Gst.State.PLAYING)
            error = self.pipeline.get_state(5000000000) 
            if error[1] != Gst.State.PLAYING:
                print("Error starting pipeline. {0}".format("") )    
                return False

        except: # GError as error:
            print("Error starting pipeline: {0}".format("unknown too"))
            raise
        return True


    def __convert_sample_to_numpy(self):
        ''' Convert a GStreamer sample to a numpy array
            Sample code from https://gist.github.com/cbenhagen/76b24573fa63e7492fb6#file-gst-appsink-opencv-py-L34
            The result is in self.img_mat.
        :return:
        '''
        self.samplelocked = True
        buf = self.sample.get_buffer()
        caps = self.sample.get_caps()
        mem = buf.get_all_memory()
        success, info = mem.map(Gst.MapFlags.READ)
        if success:
            data = info.data
            mem.unmap(info)
                
            bpp = 4
            dtype = numpy.uint8
            
            if( caps.get_structure(0).get_value('format') == "BGRx" ):
                bpp = 4

            if(caps.get_structure(0).get_value('format') == "GRAY8" ):
                bpp = 1

            if(caps.get_structure(0).get_value('format') == "GRAY16_LE" ):
                bpp = 1
                dtype = numpy.uint16

            self.img_mat = numpy.ndarray(
                (caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                bpp),
                buffer=data,
                dtype=dtype)
            self.newsample = False
            self.samplelocked = False


    def wait_for_image(self, timeout):
        ''' Wait for a new image with timeout
        :param timeout: wait time in second, should be a float number
        :return:
        '''
        tries = 10
        while tries > 0 and not self.newsample:
            tries -= 1
            time.sleep(float(timeout) / 10.0)


    def Snap_image(self, timeout):
        '''
        Snap an image from stream using a timeout.
        :param timeout: wait time in second, should be a float number. Not used
        :return: bool: True, if we got a new image, otherwise false.
        '''
        if self.ImageCallback is not None:
            print("Snap_image can not be called, if a callback is set.")
            return False

        self.wait_for_image(timeout)
        if self.sample is not None and self.newsample:
            self.__convert_sample_to_numpy()
            return True

        return False


    def Get_image(self):
        return self.img_mat


    def Stop_pipeline(self):
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_state(Gst.State.READY)
        

    def get_source(self):
        '''
        Return the source element of the pipeline.
        '''
        return self.source


    def List_Properties(self):
        property_names = self.source.get_tcam_property_names()

        for name in property_names:
            try:
                base = self.source.get_tcam_property(name)
                print("{}\t{}".format(base.get_display_name(),
                                      name
                                        ))
            except Exception as error:
                raise Exception( name + " : " + error.message )


    def Get_Property(self, PropertyName):
        """
        Return the value of the passed property.
        If something fails an
        exception is thrown.
        :param PropertyName: Name of the property to set
        :return: Current value of the property
        """
        try:
            baseproperty = self.source.get_tcam_property(PropertyName)
            val = baseproperty.get_value()
            return val

        except GLib.Error as error:
            raise Exception(PropertyName + " : " + error.message )
            return 0
        return 0


    def Set_Property(self, PropertyName, value):
        '''
        Pass a new value to a camera property. If something fails an
        exception is thrown.
        :param PropertyName: Name of the property to set
        :param value: Property value. Can be of type int, float, string and boolean
        '''
        try:
            baseproperty = self.source.get_tcam_property(PropertyName)
            baseproperty.set_value(value)
        except GLib.Error as error:
            raise Exception(PropertyName + " : " + error.message )


    def execute_command(self, PropertyName ):
        '''
        Execute a command property like Software Trigger
        If something fails an exception is thrown.
        :param PropertyName: Name of the property to set
        '''
        try:
            baseproperty = self.source.get_tcam_property(PropertyName)
            baseproperty.set_command()
        except GLib.Error as error:
            raise Exception( PropertyName + " : " + error.message )


    def Set_Image_Callback(self, function, *data):
        self.ImageCallback = function
        self.ImageCallbackData = data


    def selectDevice(self):
        ''' Select a camera, its video format and frame rate
        :return: True on success, False on nothing selected
        '''
        monitor = Gst.DeviceMonitor.new()
        monitor.add_filter("Video/Source/tcam")
        serials = []
        i = 0
        for device in monitor.get_devices():
            struc = device.get_properties()
            i +=1
            print("{} : Model: {} Serial: {} {} ".format(i, 
                                                         struc.get_string("model"),
                                                         struc.get_string("serial"),
                                                         struc.get_string("type")
                                                        ))

            serials.append("{}-{}".format(struc.get_string("serial"),
                                          struc.get_string("type")                                        )
                           )

        if i > 0:
            # i = int(input("Select : "))
            i=1
            if i == 0:
                return False

            self.serialnumber = serials[i-1]
            print(self.serialnumber)
            
            return self.selectFormat()

        return False


    def selectFormat(self):
        '''
        '''
        formats = self.createFormats()
        i = 0
        f = [] 
        for key, value in formats.items():
            f.append(key)
            i = i + 1
            print("{}: {}".format(i, key))

        # i = int(input("Select : "))
        i=1 #BGRx
        if i == 0:
            return False

        formatindex = f[i-1]
        i = 0
        for res in formats[formatindex].res_list:
            i = i + 1
            print("{}:  {}x{}".format(i, res.width, res.height))

        # i = int(input("Select : "))
        i=1 #640x480
        if i == 0:
            return False


        width = formats[formatindex].res_list[i-1].width
        height = formats[formatindex].res_list[i-1].height
        o = 0
        for rate in formats[formatindex].res_list[i-1].fps :
            o += 1
            print("{}:  {}".format(o, rate))

        framerate = formats[formatindex].res_list[i-1].fps[o-1] 
        # o = int(input("Select : "))
        o=4 # frame rate=30
        if o == 0:
            return False

        framerate = formats[formatindex].res_list[i-1].fps[o-1] 
        #print(format,width,height,framerate )
        self.openDevice(self.serialnumber, width, height, framerate, SinkFormats.BGRA, True)
        return True


    def createFormats(self):
        source = Gst.ElementFactory.make("tcambin")
        source.set_property("serial", self.serialnumber)
        
        source.set_state(Gst.State.READY)

        caps = source.get_static_pad("src").query_caps()
        format_dict = {}

        for x in range(caps.get_size()):
            structure = caps.get_structure(x)
            name = structure.get_name()
            try:
                videoformat = structure.get_value("format")

                if videoformat not in format_dict:
                    format_dict[videoformat] = FmtDesc(name, videoformat)
                    

                width = structure.get_value("width")
                height = structure.get_value("height")       

                rates = self.get_framerates(structure)
                tmprates = []

                for rate in rates:
                    tmprates.append(str(rate))

                format_dict[videoformat].res_list.append(ResDesc(width, height, tmprates))      

            except:
                print("Except")
                pass

        source.set_state(Gst.State.NULL)
        source.set_property("serial", "")
        source = None

        return format_dict

    def get_framerates(self, fmt):
        try:
            tmprates = fmt.get_value("framerate")
            if type(tmprates) == Gst.FractionRange:
                # A range is given only, so create a list of frame rate in 10 fps steps:
                rates = []
                rates.append("{0}/{1}".format(int(tmprates.start.num),int(tmprates.start.denom)))
                r = int((tmprates.start.num + 10) / 10) * 10
                while r < (tmprates.stop.num / tmprates.stop.denom ):
                    rates.append("{0}/1".format(r))
                    r += 10

                rates.append("{0}/{1}".format(int(tmprates.stop.num),int(tmprates.stop.denom)))
            else:
                rates = tmprates

        except TypeError:
            # Workaround for missing GstValueList support in GI
            substr = fmt.to_string()[fmt.to_string().find("framerate="):]
            # try for frame rate lists
            field, values, remain = re.split("{|}", substr, maxsplit=3)
            rates = [x.strip() for x in values.split(",")]
        return rates


class ResDesc:
    """"""
    def __init__(self,                 
                 width: int,
                 height: int,
                 fps: list):
        self.width = width
        self.height = height
        self.fps = fps


class FmtDesc:
    """"""

    def __init__(self,
                 name: str = "",
                 fmt: str = ""):
        self.name = name
        self.fmt = fmt
        self.res_list = []

    def get_name(self):
        if self.name == "image/jpeg":
            return "jpeg"
        else:
            return self.fmt

    def get_resolution_list(self):

        res_list = []

        for entry in self.res_list:
            res_list.append(entry.resolution)

        return res_list

    def get_fps_list(self, resolution: str):

        for entry in self.res_list:
            if entry.resolution == resolution:
                return entry.fps

    def generate_caps_string(self, resolution: str, fps: str):
        if self.name == "image/jpeg":
            return "{},width={},height={},framerate={}".format(self.name,
                                                               resolution.split('x')[0],
                                                               resolution.split('x')[1],
                                                               fps)
        else:
            return "{},format={},width={},height={},framerate={}".format(self.name,
                                                                         self.fmt,
                                                                         resolution.split('x')[0],
                                                                         resolution.split('x')[1],
                                                                         fps)
