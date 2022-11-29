#version 330 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSample;

void main()
{
  FragColor = texture(textureSample, TexCoord);
}
