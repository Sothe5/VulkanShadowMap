#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 3) uniform sampler2D texSamplerShadow;

layout(location = 0) in vec3 fragColor;	// get all the values from the vertex shader in the fragment
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragNormalCoord;

layout(location = 3) in float fragSpecularHighlight;	
layout(location = 4) in vec3 fragAmbientColor;
layout(location = 5) in vec3 fragDiffuseColor;
layout(location = 6) in vec3 fragSpecularColor;
layout(location = 7) in vec3 fragEmissiveColor;
layout(location = 8) in vec4 fragEyeVector;
layout(location = 9) in vec4 fragLightVector;
layout(location = 10) in vec4 fragShadowCoord;

layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform pushConstants {
  int isShadowMapAsTexture;
  int isDepthLightForVisible;
  int isStandardLight;
  int isAcneRemoval;
  int isPCFfiltered;
} PushConstant;

vec4 normEyeVector = normalize(fragEyeVector);
vec4 normLightVector = normalize(fragLightVector);
vec4 normNormal = normalize(fragNormalCoord);

vec3 computeAmbientLight()
{
	return fragColor * fragAmbientColor*0.2;	// compute ambient light with a coefficient of light intensity of 0.5
}

vec3 computeDiffuseLight()
{
	float diffuseDot = dot(normLightVector, normNormal);
	return fragDiffuseColor * fragColor * diffuseDot*0.5;	// compute diffuse light with a coefficient of light intensity of 0.5
}

vec3 computeSpecularLight()
{
	vec4 halfAngleVector = normalize((normEyeVector + normLightVector)/2.0);	
	float specularDot = dot(halfAngleVector, normNormal);
	float specularPower = pow(specularDot, fragSpecularHighlight);
	return fragSpecularColor * fragColor * specularPower*0.2;	// compute specular light with a coefficient of light intensity of 0.5
}

float LinearizeDepth(float depth)
{
  float n = 0.1; // camera z near
  float f = 10.0; // camera z far
  float z = depth;
  return (2.0 * n) / (f + n - z * (f - n));	
}

void main() {
	vec3 ambient = computeAmbientLight();
	vec3 diffuse = computeDiffuseLight();
	vec3 specular = computeSpecularLight();
	vec3 lightResult = ambient + diffuse + specular;

	float PCFfilterValue = 1;

	float acneRemoval = 0;
	if(PushConstant.isAcneRemoval == 1)	// if we want to remove acne
	{
		acneRemoval = 0.005;
	}

	vec4 shadowCoord = fragShadowCoord / fragShadowCoord.w;
	
	if(PushConstant.isPCFfiltered == 1)	// if we want to do PCF filter by doing the average of the neighbours shadow
	{
		ivec2 texDim = textureSize(texSamplerShadow, 0);
		float scale = 3.5;
		float dx = scale * 1.0 / float(texDim.x);	// how far do we move for each neighbour
		float dy = scale * 1.0 / float(texDim.y);

		float shadowFactor = 0.0;
		int count = 0;
		int range = 1;
		
		for (int x = -range; x <= range; x++)	// loop through neighbours
		{
			for (int y = -range; y <= range; y++)
			{
				float shadowValue = 1;
				if ( texture( texSamplerShadow, shadowCoord.xy + vec2(dx*x, dy*y)).r < shadowCoord.z - acneRemoval)  // test the depth (shadow or not in shadow)
				{
					shadowValue = ambient.x;
				}
				shadowFactor += shadowValue;
				count++;
			}
	
		}
		PCFfilterValue = shadowFactor / count;	// get the average. In the edges it will blur and in the center it will average with other shadow positions
	}
	else	// if we don't want PCF filter
	{
		if ( texture( texSamplerShadow, shadowCoord.xy ).r < shadowCoord.z - acneRemoval)  // test the depth (shadow or not in shadow)
		{
			lightResult = ambient;	// when shadow set to ambient
			if(PushConstant.isDepthLightForVisible == 1)	// if we want to see the depth of light only for visible surfaces
			{
				outColor = vec4(vec3(0),1);
			}
		}
		else
		{
			if(PushConstant.isDepthLightForVisible == 1) // if we want to see the depth of light only for visible surfaces
			{
				outColor = vec4(vec3(1.0-LinearizeDepth(shadowCoord.z)),1);	// render the depth from the light point of view
			}
		}
	}

	if(PushConstant.isShadowMapAsTexture == 0 && PushConstant.isDepthLightForVisible == 0 && PushConstant.isStandardLight == 0)	// If we want default mode
	{
		outColor = vec4(vec3( lightResult.x * PCFfilterValue,  lightResult.y * PCFfilterValue, lightResult.z * PCFfilterValue) ,1);
	}
	
	if(PushConstant.isShadowMapAsTexture == 1)	// if we want to render the shadow map as a texture
	{
		float shadow = texture( texSamplerShadow, shadowCoord.xy ).r;	// render the shadow map as a texture
		outColor = vec4(vec3(1.0-LinearizeDepth(shadow)),1);
	}
	
	if(PushConstant.isStandardLight == 1)	// if we don't want to perform shadow map
	{	
		lightResult = ambient + diffuse + specular;
		outColor = vec4(vec3( lightResult.x,  lightResult.y,  lightResult.z) ,1);
	}
	
}