$(document).ready(function() {
    var pathname = window.location.pathname;
    var cur_ver = $("#version-slug").text().replace(/^[\n\s]+|[\n\s]+$/g, '')
    if ( cur_ver.endsWith(" (dev)") ) {
        cur_ver = "master";
    }
    var relpath = pathname.substring(pathname.indexOf(cur_ver)).replace(/\/$/, '');
    var levels = relpath.split("/").length - 1
    if ( levels == 0 ) {
        levels = 1
	    relpath += "/"
    }
    var versions_file = "../".repeat(levels) + "versions.json"
    relpath = "../".repeat(levels) + relpath
    relpath = relpath.replace("//", "/")
    console.log(`relpath="${relpath}", cur_ver="${cur_ver}"`)

    $.getJSON(versions_file, function (data) {
        $("#version-slug").remove();  // Unnecessary if JSON was downloaded

        $.each(data["tags"].reverse(), function( i, val ) {
            var new_path = relpath.replace(cur_ver, val)
            var item = `<li class="toctree-l2"><a class="reference internal" href="${new_path}">${val}</a></li>`
            console.log(item)
            $("#v-tags").append(item)
        });
    }).fail(function() {
      $("#version-menu").hide();  // JSON download failed - hide dropdown
    });
});