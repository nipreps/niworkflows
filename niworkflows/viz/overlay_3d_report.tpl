<div id='{{unique_string}}'>
    <style type='text/css'>
        @keyframes flickerAnimation {
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
            animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation;
        }
    </style>
    <h4>{{title}}</h4>
    <div class='image_container'>
        <div class='base_image'>{{base_image}}</div>
        <div class='overlay_image'>{{overlay_image}}</div>
    </div>
</div>
