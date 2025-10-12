# Image Management System

This directory contains the image management system for PriceOptima, supporting multiple image sources and local file management.

## Directory Structure

```
public/images/
├── testimonials/          # Testimonial images
├── features/             # Feature section images
├── backup/               # Backup images
├── image-config.json     # Image configuration
└── README.md            # This file
```

## Adding Your Own Images

### 1. Testimonial Images
Place your testimonial images in the `testimonials/` directory with these naming conventions:

- `adebayo-ogundimu.jpg` - Nigerian male, Lagos
- `chioma-okwu.jpg` - Dark Igbo lady, Port Harcourt
- `grace-okafor.jpg` - Fair Igbo lady, Abuja
- `emeka-nwosu.jpg` - Nigerian male CEO, Kano
- `amina-hassan.jpg` - Black Fulani lady, Kaduna
- `kwame-asante.jpg` - Black African man from Kumasi, Ghana
- `tunde-adebayo.jpg` - Nigerian male, IT Manager

### 2. Feature Images
Place feature images in the `features/` directory:

- `smart-pricing.jpg` - Smart pricing in action
- `profit-optimization.jpg` - Profit optimization dashboard
- `ai-analytics.jpg` - AI-powered analytics

### 3. Image Specifications

**Testimonial Images:**
- Size: 150x150 pixels (square)
- Format: JPG or PNG
- Quality: High resolution for web
- Style: Professional headshots

**Feature Images:**
- Size: 400x300 pixels (landscape)
- Format: JPG or PNG
- Quality: High resolution for web
- Style: Professional business/technology images

## Image Sources

The system supports multiple image sources:

1. **Pexels** (Primary) - High-quality free stock photos
2. **Unsplash** (Fallback) - Alternative free stock photos
3. **Local Files** - Your own custom images

## Configuration

Edit `image-config.json` to:
- Switch between image sources
- Add new image alternatives
- Update image metadata
- Enable/disable local images

## Usage in Code

```typescript
import { getTestimonialImage, getImageWithFallback } from '@/lib/image-manager';

// Get primary image
const imageUrl = getTestimonialImage('adebayo-ogundimu');

// Get image with fallbacks
const { primary, fallbacks } = getImageWithFallback('chioma-okwu', 'testimonials');
```

## Image Services

### Pexels
- URL: `https://images.pexels.com/photos/[id]/pexels-photo-[id].jpeg`
- Quality: High
- License: Free for commercial use

### Unsplash
- URL: `https://images.unsplash.com/photo-[id]`
- Quality: High
- License: Free for commercial use

## Backup Strategy

1. Keep original images in `backup/` directory
2. Maintain multiple versions of each image
3. Document image sources and licenses
4. Regular backups of image configuration

## Troubleshooting

### Images Not Loading
1. Check if local images exist in correct directory
2. Verify image-config.json is valid
3. Check network connectivity for remote images
4. Ensure image URLs are accessible

### Image Quality Issues
1. Use high-resolution source images
2. Optimize images for web (compress but maintain quality)
3. Ensure proper aspect ratios
4. Test on different screen sizes

## Best Practices

1. **Naming Convention**: Use kebab-case for filenames
2. **File Organization**: Keep images in appropriate subdirectories
3. **Optimization**: Compress images for web performance
4. **Backup**: Always keep backups of important images
5. **Documentation**: Update image-config.json when adding new images
6. **Testing**: Test images on different devices and browsers

## Adding New Images

1. Add image file to appropriate directory
2. Update `image-config.json` with new image details
3. Test the image loads correctly
4. Update any hardcoded references in code
5. Commit changes to version control

## Support

For image-related issues:
1. Check this README first
2. Verify image-config.json syntax
3. Test image URLs manually
4. Check browser console for errors
5. Contact development team if issues persist
