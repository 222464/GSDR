uniform sampler2D texture;

float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

void main()
{
    // lookup the pixel in the texture
    vec4 pixel = texture2D(texture, gl_TexCoord[0].xy);

    // multiply it by the color
	float c = pixel.x * 2.0 - 1.0;
	
    gl_FragColor = gl_Color * min(1.0, max(0.0, 1.01 * sigmoid(100.0 * (c + 0.3))));
    //gl_FragColor = pixel.x > 0.7 ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}