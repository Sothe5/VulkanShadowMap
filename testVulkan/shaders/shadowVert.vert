#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {	// Model View Projection matrices
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;
	
layout(binding = 1) uniform UniformBufferLight {	// values of the material related to lighting
    float specularHighlight;
	vec3 ambientColor;
	vec3 diffuseColor;
	vec3 specularColor;
	vec3 emissiveColor;
	vec4 lightPosition;
} ubl;

layout(binding = 2) uniform UniformBufferShadow {	// values of the material related to shadows
   mat4 view;
   mat4 proj;
} ubs;

layout(location = 0) in vec3 inPosition;	// position
layout(location = 1) in vec3 inColor;	// color
layout(location = 2) in vec2 inTexCoord;	// textCoord
layout(location = 3) in vec3 inNormalCoord;	// normal

void main() {
	vec4 WCS_position = ubo.model * vec4(inPosition, 1.0);	// Position in VCS

	vec4 LCS_position = ubs.view * WCS_position;	// position in LCS

    gl_Position = ubs.proj * LCS_position;	// position in SCS using the view and projection of the light
}