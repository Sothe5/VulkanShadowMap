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


layout(location = 0) out vec3 fragColor;	// variables that will be passed to the fragment shader
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec4 fragNormalCoord;

layout(location = 3) out float fragSpecularHighlight;
layout(location = 4) out vec3 fragAmbientColor;
layout(location = 5) out vec3 fragDiffuseColor;
layout(location = 6) out vec3 fragSpecularColor;
layout(location = 7) out vec3 fragEmissiveColor;
layout(location = 8) out vec4 fragEyeVector;
layout(location = 9) out vec4 fragLightVector;
layout(location = 10) out vec4 fragShadowCoord;

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

void main() {

	vec4 WCS_position = ubo.model * vec4(inPosition, 1.0);	// Position in VCS

	vec4 VCS_position = ubo.view * WCS_position;	// Position in VCS
	
	vec4 LCS_position = ubs.view * WCS_position;

    fragShadowCoord = biasMat * ubs.proj * LCS_position;	// coordinates from that vertex from the light

    gl_Position = ubo.proj * VCS_position;	// position in NVCS
    fragColor = inColor;
    fragTexCoord = inTexCoord;
	fragNormalCoord = ubo.view * ubo.model * vec4(inNormalCoord,0.0);	// normal in VCS
	fragEyeVector = -1 * VCS_position;	// vector from the pixel to the eye
	fragLightVector =  ubo.view * ubl.lightPosition - VCS_position;	// vector from the light to the pixel

	fragSpecularHighlight = ubl.specularHighlight;	// lighting values
	fragAmbientColor = ubl.ambientColor;
	fragDiffuseColor = ubl.diffuseColor;
	fragSpecularColor = ubl.specularColor;
	fragEmissiveColor = ubl.emissiveColor;
}