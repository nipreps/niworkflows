<div id='{{unique_string}}'>
    <h4>{{title}}</h4>
    <style type='text/css'>
    @keyframes flickerAnimation{{unique_string}} {
        0% {
            opacity: 1;
        }
        100% {
            opacity: 0;
        }
    }

    #{{unique_string}} .image_container {
        position: relative;
    }

    #{{unique_string}} .image_container .overlay_image {
        position: absolute;
            top: 0;
            left: 0;
        background-size: 100%;
        animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation{{unique_string}};
    }

    #{{unique_string}} .image_container .overlay_image:hover {
        animation-play-state: paused;
    }
    </style>
    <div class='image_container'>
        <div class='base_image'>{{base_image}}</div>
        <div class='overlay_image'>{{overlay_image}}</div>
    </div>
    <h5>Inputs</h5>
    <p><pre>{{inputs}}</pre></p>
    <h5>Outputs</h5>
    <p><pre>{{outputs}}</pre></p>
    <p> Generated with overlay_3d_report.tpl </p>
</div>
