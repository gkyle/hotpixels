local LrApplication = import("LrApplication")
local LrDate        = import("LrDate")
local LrErrors      = import("LrErrors")
local LrFileUtils   = import("LrFileUtils")
local LrPathUtils   = import("LrPathUtils")
local LrLogger      = import("LrLogger")
local myLogger      = LrLogger("hotpixels")

myLogger:enable("logfile")

Util = {}

-- find path from a text file output each time the app runs
Util.findPath = function()
    pathPath = LrPathUtils.standardizePath(_PLUGIN.path.."/PluginPath.txt")
    myLogger:trace(string.format("Util.processRenderedPhotos pathPath: %s", pathPath))
    if LrFileUtils.exists(pathPath) then
        appPath = LrFileUtils.readFile(pathPath)
        if LrFileUtils.exists(appPath) then
            return true, appPath
        else
            errmsg = "Could not find path to application: "..appPath.."."
        end
    else
        errmsg = "Could not find path to application. Try running Hot Pixels directly, then try again."
    end
    myLogger:trace(errmsg)
    LrErrors.throwUserError(errmsg)
    return false, nil
end

Util.makeTempDir = function()
	local parentTempPath = LrPathUtils.getStandardFilePath("temp")
	local tempPath = nil
	repeat
	do
		local now = LrDate.currentTime()
		tempPath = LrPathUtils.child(parentTempPath, "Lightroom_Export_"..LrDate.timeToUserFormat(now, "%Y%m%d%H%M%S"))
		if LrFileUtils.exists(tempPath) then
			tempPath = nil
		else
			LrFileUtils.createAllDirectories(tempPath)
		end
	end until tempPath
	myLogger:trace(string.format("tempPath: %s", tempPath))

    return tempPath
end

Util.copyFile = function(srcPath, destPath)
    local success, message

    myLogger:trace(string.format("srcPath: %s", srcPath))
    myLogger:trace(string.format("dstPath: %s", destPath))
    success, message = LrFileUtils.copy(srcPath, destPath)

    if success then
        return true
    end

    if message == nil then
        message = " (reason unknown)"
    end
    myLogger:trace("Unable to copy file "..srcPath..message)
    LrErrors.throwUserError("Unable to copy: "..srcPath..message)

    return false
end

-- When app returns, copy resulting files to catalog
Util.onFinish = function(status, tempPath, sourcePhoto, inputFiles)
	myLogger:trace("Finished. Syncing files back to LRC.")
	local catalog = LrApplication.activeCatalog()

	for filename in LrFileUtils.files(tempPath) do
		myLogger:trace(string.format("filename: %s", filename))
        if not inputFiles[filename] then
			srcPath = sourcePhoto:getRawMetadata("path")
			local copyPath = LrPathUtils.child(LrPathUtils.parent(srcPath), LrPathUtils.leafName(filename))
			myLogger:trace(string.format("Copy to LRC: srcPath: %s", filename))
			myLogger:trace(string.format("Copy to LRC: dstPath: %s", copyPath))

            -- Copy file into catalog
			catalog:withWriteAccessDo("Copying file", function(context)
				success, message = LrFileUtils.copy(filename, copyPath)
				if success then
					local photo = catalog:addPhoto(copyPath, sourcePhoto, "above")
					for i, field in ipairs(METADATA_FIELDS) do
						local value = sourcePhoto:getFormattedMetadata(field)
						if value ~= nil then
							myLogger:trace(string.format("Setting %s to %s", field, value))
							photo:setRawMetadata(field, value)
						end
					end
				else
					myLogger:trace("Unable to copy file ")
					myLogger:trace(message)
				end				
			end)
        end
	end
end

Util.exportFiles = function(exportContext, tempPath, maxFiles)
    local exportSession = exportContext.exportSession

	-- We export the first file in selection
	if exportSession:countRenditions() > 0 then
		local progressScope
		progressScope = exportContext:configureProgress({
			title = LOC(string.format("$$$/Hot Pixels/Upload/Progress=Exporting %d photos to Hot Pixels", exportSession:countRenditions()))
		})

		-- export
		local fileCount = 0
		for i, rendition in exportContext:renditions { stopIfCanceled = true } do
			local success, pathOrMessage = rendition:waitForRender()
			sourcePath = rendition.photo:getRawMetadata('masterPhoto'):getRawMetadata('path')
			if progressScope:isCanceled() then
				break
			end

			if not success then
				myLogger:trace("Unable to export: "..pathOrMessage)
				LrErrors.throwUserError("Unable to export: "..pathOrMessage)
				LrFileUtils.delete(tempPath)
				return
			end

			fileCount = fileCount + 1
			if fileCount > maxFiles then
				break
			end

			-- copy
			local basename = LrPathUtils.leafName(rendition.destinationPath)
			local copyPath = LrPathUtils.child(tempPath, basename)
			local success = Util.copyFile(rendition.destinationPath, copyPath)

			if progressScope:isCanceled() then
				LrFileUtils.delete(tempPath)
				return
			end
			if not success then
				myLogger:trace("Unable to export: "..pathOrMessage)
				LrErrors.throwUserError("Unable to export: "..pathOrMessage)
				LrFileUtils.delete(tempPath)
				return
			end
		end
	end
end

Util.getCommand = function(paramString)
	local command = appPath..' '..paramString..' &'
	if WIN_ENV then
		-- For Windows, show the command window
		command = '"start /wait cmd.exe /c '..appPath..' '..paramString..'"'
	elseif MAC_ENV then
		-- For Mac, use osascript to open a new Terminal window
		command = 'osascript -e '..'tell application "Terminal" to do script "'..appPath..' '..paramString..'"'
	end

    return command
end