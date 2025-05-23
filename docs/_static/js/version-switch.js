$(document).ready(function() {
    const pathname = window.location.pathname;
    const cur_ver = $("#version-slug").text().replace(/^[\n\s]+|[\n\s]+$/g, '')
    let major_minor = "master";
    if ( cur_ver.lastIndexOf(" (dev)") == -1 ) {
        major_minor = `${cur_ver.split('.')[0]}.${cur_ver.split('.')[1]}`
    }
    let relpath = pathname.substring(pathname.lastIndexOf(major_minor)).replace(/\/$/, '');
    let levels = relpath.split("/").length - 1
    if ( levels == 0 ) {
        levels = 1
        relpath += "/"
    }
    const versions_file = "../".repeat(levels) + "versions.json"
    relpath = "../".repeat(levels) + relpath
    relpath = relpath.replace("//", "/")
    console.log(`relpath="${relpath}", cur_ver="${cur_ver}"`)

    $.getJSON(versions_file, function (data) {
        $("#version-slug").remove();  // Unnecessary if JSON was downloaded

        $.each(data["tags"].reverse(), function( i, val ) {
            const new_path = relpath.replace(major_minor, val)
            let item = `<li class="toctree-l2"><a class="reference internal" href="${new_path}">${val}</a></li>`
            if ( i == 0 ) {
                item = `<li class="toctree-l2"><a class="reference internal" href="${new_path}">${val} (Latest Release)</a></li>`
            }
            $("#v-tags").append(item)
        });
        $.each(data["heads"].reverse(), function( i, val ) {
            const new_path = relpath.replace(major_minor, val)
            let item = `<li class="toctree-l2"><a class="reference internal" href="${new_path}">${val}</a></li>`
            if ( ["master", "main"].includes(val) ) {
                item = `<li class="toctree-l2"><a class="reference internal" href="${new_path}">${val} (Development)</a></li>`
            }
            $("#v-tags").append(item)
        });
    }).fail(function() {
      $("#version-menu").hide();  // JSON download failed - hide dropdown
    });
});
