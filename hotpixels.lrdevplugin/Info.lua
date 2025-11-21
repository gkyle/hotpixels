return {
    VERSION = { major=0, minor=1, revision=4, },
  
    LrSdkVersion = 14.2,
    LrSdkMinimumVersion = 4.0,
  
    LrToolkitIdentifier = "com.github.gkyle.hotpixels",
    LrPluginName = "Hot Pixels",
    LrPluginInfoUrl="https://www.github.com/gkyle/hotpixels",

	LrExportMenuItems = {
        {
            title = "Correct Images with Hot Pixels",
            file = "ImageExport.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = "Generate a Profile from Dark Images",
            file = "DarkFrameExport.lua",
            enabledWhen = "photosSelected",
        },
	},

	LrExportServiceProvider = {
        title = "Hot Pixels",
        file = "ExportServiceProvider.lua",
		builtInPresetsDir = "presets",
	},
  }