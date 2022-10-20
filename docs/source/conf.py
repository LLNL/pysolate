# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# LLNL-CODE-841231
# author: Andrea Chiang (andrea@llnl.gov)
#
# pysolate documentation build configuration file.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import glob
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'pysolate'
copyright = '2021, Lawrence Livermore National Security, LLC'
author = 'Ana Aguiar and Andrea Chiang'

# The short X.Y version.
import pysolate
from packaging.version import parse
parsed_version = parse(pysolate.__version__)
version = ".".join(parsed_version.base_version.split(".")[:2])
# The full version, including alpha/beta/rc tags
if parsed_version.is_postrelease:
    release = parsed_version.base_version
else:
    release = pysolate.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
#master_doc = 'contents'

# The encoding of source files.
#source_encoding = 'utf-8'

# Automatically generate stubs
autosummary_generate = glob.glob("reference" + os.sep + "*.rst")

# This value is a list of autodoc directive flags that should be automatically
# applied to all autodoc directives. The supported flags are 'members',
# 'undoc-members', 'private-members', 'special-members', 'inherited-members' and
# 'show-inheritance'.
autodoc_default_flags = ['show-inheritance']

#This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'),
# by member type (value 'groupwise') or by source order (value 'bysource').
# The default is alphabetical.
autodoc_member_order = "bysource"

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'literal'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# Don't merge __init__ method in auoclass content
autoclass_content = 'class'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['pysolate.']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('http://matplotlib.org/', None),
    'obspy': ('https://docs.obspy.org/', None),
    'pyasdf': ('https://seismicdata.github.io/pyasdf/', None),
}

# warn about *all* references where the target cannot be found
nitpicky = True
nitpick_ignore = [('py:class', 'type')]

# -- Options for HTML output -------------------------------------------------

import sphinx_rtd_theme

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'pysolate'

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': project,

    # Tab name for entire site. (Default: "Site")
    'navbar_site_name': "Site",

    # A list of tuples containing pages or urls to link to.
    # Valid tuples should be in the following forms:
    #    (name, page)                 # a link to a page
    #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
    #    (name, "http://example.com", True) # arbitrary absolute url
    # Note the "1" or "True" value above as the third argument to indicate
    # an arbitrary url.
    'navbar_links': [
    #    ("Examples", "examples"),
        ("GitHub", "https://github.com/LLNL/pysolate", True),
    ],

    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': True,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': True,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Page",

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 2,

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden': "true",

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    'navbar_class': "navbar navbar-inverse",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top': "true",

    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    'source_link_position': "nav",

}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = { 'logo_only' : True }

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'OS-logo-horizontal-white.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'OS-icon-color.png'

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'pysolatedoc'

# If true, the reST sources are included in the HTML build as _sources/name.
html_copy_source = True
