# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# LLNL-CODE-841231
# authors:
#        Andrea Chiang (andrea@llnl.gov)
"""
Routines for handling waveform data
"""


import obspy


class Waveforms(object):
    """
    Main class for waveform data

    :param tag_options: a list of available tags, this defines the list of
        keys for the ``data`` attribute.
    :type tag_options: list

    .. rubric:: Additional Attributes

    ``station_name`` : list of str
        stations names.
    ``data`` : dict of ObsPy Stream objects
        waveform data, refer to :meth:`obspy.core.stream.Stream.select`
        for details on how to query the traces.
    """
    def __init__(self, tag_options):
        self.tag_options = tag_options
        self.station_name = []
        self.data = {}

    def add_waveform(self, wfs, tag):
        """
        Function to add seismic waveform data

        :param wfs: waveform files.
        :type wfs: :class:`~obspy.core.stream.Stream`,
            :class:`~obspy.core.trace.Trace`, str, ...
        :param tag: data type.
        :type tag: str
        """
        if tag not in self.tag_options:
            msg = "Invalid waveform data type, tag options are %s" %self.tag_options
            raise ValueError(msg)

        wfs = self._parse_waveform(wfs)

        for trace in wfs:
            info = self._collect_trace_info(trace)
            self._add_station_to_list(info["station_name"])

        self.data[tag] = wfs

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
