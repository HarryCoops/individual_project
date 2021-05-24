import os

FUNC_NAME_COLOR = "#598bb7"
BACKGROUND_COLOR = "#ffffff"
FILENAME_COLOR = "#6b6565"
GROUP_BUTTON_BG_COLOR = "#ffffff"
GROUP_BUTTON_TEXT_COLOR = "#6b6565"
FONT_SIZE = "15px;font-weight:600;"


def edit_profile_html(html):
	# change bg color
	before, after = html.split('body,html{background-color:')
	html = before + 'body,html{background-color:' + BACKGROUND_COLOR + after[7:]

	# change func name color
	before, after = html.split(';cursor:default}.application-code .name[data-v-dcd8e382]{color:')
	html = before + ';cursor:default}.application-code .name[data-v-dcd8e382]{color:' + FUNC_NAME_COLOR + after[7:]
	
	# change filename colour
	before, after = html.split('code-position[data-v-dcd8e382]{color:')
	html = before + 'code-position[data-v-dcd8e382]{color:' + FILENAME_COLOR + after[18:]
	
	# change button bg color
	before, after = html.split(
		'group-header-button[data-v-dcd8e382]:before{position:absolute;left:-3px;right:-3px;top:0;bottom:-1px;content:"";z-index:-1;background-color:'
	)
	html = ( before + 
		'group-header-button[data-v-dcd8e382]:before{position:absolute;left:-3px;right:-3px;top:0;bottom:-1px;content:"";z-index:-1;background-color:'
	+ GROUP_BUTTON_BG_COLOR + after[7:])

	# change button text color
	before, after = html.split(".group-header-button[data-v-dcd8e382]{display:inline-block;color:")
	html = before + ".group-header-button[data-v-dcd8e382]{display:inline-block;color:" + GROUP_BUTTON_TEXT_COLOR + after[19:]

	
	# change font size 
	before, after = html.split("ource Code Pro,Roboto Mono,Consolas,Monaco,monospace;font-size:")
	html = before + "ource Code Pro,Roboto Mono,Consolas,Monaco,monospace;font-size:" + FONT_SIZE + after[3:]

	return html


if __name__ == "__main__":
	root = "."
	profile_files = []
	for path, subdirs, files in os.walk(root):
	    for name in files:
	    	if name == "profile.html":
	        	profile_files.append(os.path.join(path, name))

	for profile_file in profile_files:
		with open(profile_file) as f:
			profile_html = f.read()
		if len(profile_html) == 0:
			continue
		edited_profile = edit_profile_html(profile_html)
		path = profile_file.split("profile.html")[0]
		with open(path + "edited_profile.html", "w") as f:
			f.write(edited_profile)
