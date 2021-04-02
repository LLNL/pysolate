# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-XXXXX
# authors:
#        Andrea Chiang (andrea4@llnl.gov)
"""
Routines for handling waveform data
"""


import obspy


class Waveforms(object):
    def __init__(self, tag_options):
        self.station_name = []
        self.data = {}
        self.tag_options = tag_options

    def add_waveform(self, wfs, tag):
        """
        Function to add seismic waveform data.
        """
        if tag not in self.tag_options:
            msg = "Invalid waveform data type, tag options are %s" %self.tag_options
            raise ValueError(msg)

        wfs = self._parse_waveform(wfs)

        for trace in wfs:
            info = self._collect_trace_info(trace)
            self._add_station_to_list(info["station_name"])

        self.data[tag] = wfs

    def get_data(self, tag, **kwargs):
        """
        Return a new Stream object that contains traces matching the given stats criteria,
        the new traces are only references to the original traces.
        Refer to :meth:`~obspy.core.stream.Stream.select()` for acceptable kwargs.
        """
        return self.data[tag].select(**kwargs)

    def _parse_waveform(self, wfs):
        if isinstance(wfs, obspy.Trace):
            wfs = obspy.Stream(traces=[wfs])
        elif isinstance(wfs, obspy.Stream):
            pass
        else:
            wfs = obspy.read(wfs) # attempt to read file using ObsPy

        return wfs

    def _collect_trace_info(self, trace):
        # Only station name is collected right now
        station_name = "%s.%s.%s"%(
            trace.stats.network,
            trace.stats.station,
            trace.stats.location
        )

        info = {"station_name": station_name}

        return info

    def _add_station_to_list(self, station_name):
        if station_name not in self.station_name:
            self.station_name.append(station_name)
