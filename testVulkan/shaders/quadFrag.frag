#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 3) uniform sampler2D texSamplerShadow;

layout(location = 1) in vec2 fragTexCoord;


layout(location = 0) out vec4 outColor;

float LinearizeDepth(float depth)
{
  float n = 0.1; // camera z near
  float f = 10.0; // camera z far
  float z = depth;
  return (2.0 * n) / (f + n - z * (f - n));	
}

void main() {
	vec4 textureSampled = texture(texSamplerShadow, fragTexCoord);	// sample the shadow texture
	
	outColor = vec4(vec3(1.0-LinearizeDepth(textureSampled.x)), 1); // we linearize the depth depending on the near and far plane so it can be visible
}