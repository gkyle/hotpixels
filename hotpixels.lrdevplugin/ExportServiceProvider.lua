local LrApplication   = import("LrApplication")
local LrExportSession = import("LrExportSession")
local LrErrors        = import("LrErrors")
local LrFileUtils     = import("LrFileUtils")
local LrPathUtils     = import("LrPathUtils")
local LrDate          = import("LrDate")
local LrShell         = import("LrShell")
local LrTasks         = import("LrTasks")
local LrLogger        = import("LrLogger")
local myLogger        = LrLogger("hotpixels")

myLogger:enable("logfile")

require("Util")

-- metadata fields to copy from exported source to imported copy
METADATA_FIELDS = {
	"rating",
	"dateCreated",
}

processRenderedPhotos = function(functionContext, exportContext)
	-- Get path to app
	success, appPath = Util.findPath()
	if not success then
		return
	end

	-- Use a temporary directory to pass files in/out of app
	local tempPath = Util.makeTempDir()
	local catalog = LrApplication.activeCatalog()
	local sourcePhoto = catalog:getTargetPhoto()

    -- Copy Local Files to Temp Dir
	Util.exportFiles(exportContext, tempPath, HP_commandMaxFiles)

	inputFiles = {}
	str_filenames = ''
	for filename in LrFileUtils.files(tempPath) do
		inputFiles[filename] = true
		str_filenames = str_filenames..' "'..filename..'"'
	end
    commandLineModeFlag = HP_commandLineMode
	command = Util.getCommand(commandLineModeFlag..' '..str_filenames)

	-- Run async
	LrTasks.startAsyncTask(function ()
		myLogger:trace(command)
		local ret = LrTasks.execute(command)
		Util.onFinish(ret, tempPath, sourcePhoto, inputFiles)
	end)
end


return {
	showSections = { 'fileNaming', 'fileSettings', 'imageSettings', 'metadata' },
	allowFileFormats = { 'DNG' },
	allowColorSpaces = { 'sRGB' },
	hidePrintResolution = false,
	exportPresetFields = { key = 'path', default = '' },
	processRenderedPhotos = processRenderedPhotos,
}
