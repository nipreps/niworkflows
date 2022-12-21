$(document).ready(function() {
    var pathname = window.location.pathname;
    var cur_ver = $("#version-slug").text().replace(/^[\n\s]+|[\n\s]+$/g, '')
    var major_minor = "master";
    if ( cur_ver.lastIndexOf(" (dev)") == -1 ) {
        major_minor = `${cur_ver.split('.')[0]}.${cur_ver.split('.')[1]}`
    }
    var relpath = pathname.substring(pathname.lastIndexOf(major_minor)).replace(/\/$/, '');
    var levels = relpath.split("/").length - 1
    if ( levels == 0 ) {
        levels = 1
      relpath += "/"
    }
    var versions_file = "../".repeat(levels) + "versions.json"
    relpath = "../".repeat(levels) + relpath
    relpath = relpath.replace("//", "/")

    $.getJSON(versions_file, function (data) {
        $.each(data["tags"].reverse(), function( i, val ) {
            var new_path = relpath.replace(major_minor, val)
            var item = `<dd><a href="${new_path}">${val}</a></dd>`
            $("#v-tags").append(item)
        });
        $.each(data["heads"], function( i, val ) {
            var new_path = relpath.replace(major_minor, val)
            var item = `<dd><a href="${new_path}">${val}</a></dd>`
            $("#v-branches").append(item)
        });
    });
    SphinxRtdTheme.Navigation.enable(true)
});