
use clap::{Parser, ValueEnum};
use image::RgbImage;
use rayon::prelude::*;
use std::f32::consts::{PI, TAU};
use std::path::PathBuf;
use std::time::Instant;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions, KeyRepeat};

// ===== Configurables =====
pub const SUN_LIGHT_INTENSITY: f32 = 2.4; // Intensidad fija del Sol
pub const ORBIT_MODE_DEFAULT: bool = true; // Modo órbita por defecto
pub const ORBIT_SCALE: f32 = 2.8;         // Escala de órbitas (distancia del centro)
pub const AMBIENT_BASE: f32 = 0.22;       // Iluminación ambiente mínima
pub const RIM_INTENSITY: f32 = 0.18;      // Intensidad del rim light
pub const AUTO_SPIN_BASE: f32 = 0.45;     // Giro automático en vista individual (rad/s)

// --- Luna del planeta rocoso (RockyCratered) ---
pub const ROCKY_MOON_ENABLED_SYSTEM: bool = true;
pub const ROCKY_MOON_ENABLED_SINGLE: bool = true;
pub const ROCKY_MOON_SCALE: f32 = 0.28;   // tamaño relativo a su planeta
pub const ROCKY_MOON_DIST: f32  = 1.9;    // distancia en radios del planeta
pub const ROCKY_MOON_TILT: f32  = 0.35;   // elevación en radios
pub const ROCKY_MOON_SPEED: f32 = 1.15;   // velocidad orbital (rad/s)

// Top 6 más complejos (para galería)
pub const TOP6: [Body; 6] = [
    Body::Sun,
    Body::OceanWorld,
    Body::Volcanic,
    Body::RockyCratered,
    Body::GasBands,
    Body::GasRings,
];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Body {
    Sun,
    RockyCratered,
    RockyDesert,
    OceanWorld,
    Volcanic,
    GasBands,
    GasRings,
    IceSoft,
    IceTurbulent,
    System,
    All,
}

#[derive(Parser, Debug)]
#[command(name="proc_solar", about="Software renderer: Sun + planetas (procedural shaders, no textures).")]
struct Opts {
    #[arg(short, long, value_enum, default_value_t=Body::System)]
    body: Body,
    #[arg(short, long, default_value = "out")]
    out: PathBuf,
    #[arg(short='W', long, default_value_t=1280)]
    width: u32,
    #[arg(short='H', long, default_value_t=800)]
    height: u32,
    #[arg(short, long, default_value_t=0.0)]
    time: f32,
    #[arg(long, default_value_t=false)]
    hq: bool,
    /// Ventana interactiva
    #[arg(long, default_value_t=false)]
    window: bool,
    /// FPS objetivo (ventana)
    #[arg(long, default_value_t=30)]
    fps: u32,
    /// Animar órbitas (ventana)
    #[arg(long, default_value_t=true)]
    orbits: bool,
    /// Galería Top6 (muestra 1 cuerpo a la vez y puedes ciclar con [ ])
    #[arg(long, default_value_t=false)]
    gallery_top6: bool,
}

#[inline] fn clamp01(x:f32)->f32{ x.max(0.0).min(1.0) }
#[inline] fn lerp(a:f32,b:f32,t:f32)->f32{ a + (b-a)*t }
#[inline] fn smoothstep(a:f32,b:f32,x:f32)->f32{
    let t = clamp01((x-a)/(b-a)); t*t*(3.0-2.0*t)
}
#[inline] fn fract(x:f32)->f32{ x - x.floor() }

#[inline] fn hash(mut x:u32)->u32{
    x ^= x >> 16; x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15; x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16; x
}
#[inline] fn rng2(ix:i32, iy:i32) -> f32 {
    let h = hash(((ix as u32) << 16) ^ (iy as u32) ^ 0x9e37_79b9);
    (h as f32) / (u32::MAX as f32)
}
fn noise2(x:f32, y:f32) -> f32 {
    let xi = x.floor() as i32; let yi = y.floor() as i32;
    let xf = x - xi as f32; let yf = y - yi as f32;
    let s = rng2(xi, yi);
    let t = rng2(xi+1, yi);
    let u = rng2(xi, yi+1);
    let v = rng2(xi+1, yi+1);
    let sx = xf*xf*(3.0-2.0*xf);
    let sy = yf*yf*(3.0-2.0*yf);
    let a = s + (t - s)*sx;
    let b = u + (v - u)*sx;
    a + (b - a)*sy
}
fn fbm2(x:f32, y:f32, octaves:i32, lacunarity:f32, gain:f32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for _ in 0..octaves {
        sum += amp * noise2(x*freq, y*freq);
        freq *= lacunarity; amp *= gain;
    }
    sum
}

#[derive(Clone, Copy, Debug)]
struct Ray{ o:Vec3, d:Vec3 }
#[derive(Clone, Copy, Debug)]
struct Hit{ t:f32, n:Vec3, uv:(f32,f32), id:usize }

#[derive(Clone, Copy, Debug)]
struct Vec3{ x:f32, y:f32, z:f32 }
impl Vec3{
    #[inline] fn new(x:f32,y:f32,z:f32)->Self{ Self{x,y,z} }
    #[inline] fn add(self, o:Self)->Self{ Self::new(self.x+o.x, self.y+o.y, self.z+o.z) }
    #[inline] fn sub(self, o:Self)->Self{ Self::new(self.x-o.x, self.y-o.y, self.z-o.z) }
    #[inline] fn mul(self, s:f32)->Self{ Self::new(self.x*s, self.y*s, self.z*s) }
    #[inline] fn dot(self, o:Self)->f32{ self.x*o.x + self.y*o.y + self.z*o.z }
    #[inline] fn cross(self, o:Self)->Self{
        Self::new(self.y*o.z - self.z*o.y, self.z*o.x - self.x*o.z, self.x*o.y - self.y*o.x)
    }
    #[inline] fn len(self)->f32{ self.dot(self).sqrt() }
    #[inline] fn norm(self)->Self{ let l=self.len(); Self::new(self.x/l, self.y/l, self.z/l) }
}

#[derive(Clone, Copy, Debug)]
struct Camera{
    pos: Vec3,
    yaw: f32,    // izquierda/derecha (rad). 0 mira a -Z
    pitch: f32,  // arriba/abajo (rad)
    fov_deg: f32,
}
impl Camera{
    fn forward(&self)->Vec3{
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        Vec3::new(sy*cp, sp, -cy*cp).norm()
    }
    fn basis(&self, aspect:f32)->(Vec3,Vec3,Vec3,f32){
        let f = self.forward();
        let world_up = Vec3::new(0.0,1.0,0.0);
        let r = world_up.cross(f).norm();
        let u = f.cross(r).norm();
        let scale = (0.5*self.fov_deg.to_radians()).tan();
        (f,r,u,scale)
    }
}

#[derive(Clone, Copy, Debug)]
struct Orbit {
    enabled: bool,
    target: Vec3,
    distance: f32,
}

#[derive(Clone, Copy, Debug)]
struct Sphere {
    c: Vec3,
    r: f32,
    body: Body,
    rot_speed: f32, // rad/s
}

#[derive(Clone, Copy, Debug)]
struct Ring {
    c: Vec3,
    n: Vec3,
    r_inner: f32,
    r_outer: f32,
}

#[derive(Clone, Debug)]
struct Scene {
    spheres: Vec<Sphere>,
    rings: Vec<Ring>,
}

fn intersect_sphere(ray:&Ray, s:&Sphere) -> Option<Hit>{
    let oc = ray.o.sub(s.c);
    let b = oc.dot(ray.d);
    let c = oc.dot(oc) - s.r*s.r;
    let disc = b*b - c;
    if disc < 0.0 { return None; }
    let t1 = -b - disc.sqrt();
    let t2 = -b + disc.sqrt();
    let t = if t1>1e-3 { t1 } else if t2>1e-3 { t2 } else { return None; };
    let p = ray.o.add(ray.d.mul(t));
    let n = p.sub(s.c).mul(1.0/s.r);
    let u = 0.5 + n.z.atan2(n.x)/TAU;
    let v = 0.5 - n.y.asin()/PI;
    Some(Hit{ t, n:n.norm(), uv:(u,v), id:0 })
}

fn intersect_ring(ray:&Ray, r:&Ring) -> Option<(f32, f32)> {
    let denom = ray.d.dot(r.n);
    if denom.abs() < 1e-4 { return None; }
    let t = (r.c.sub(ray.o)).dot(r.n) / denom;
    if t < 1e-3 { return None; }
    let p = ray.o.add(ray.d.mul(t));
    let rel = p.sub(r.c);
    let dist = rel.len();
    if dist >= r.r_inner && dist <= r.r_outer { Some((t, dist)) } else { None }
}

fn palette(a:(f32,f32,f32), b:(f32,f32,f32), t:f32)->(f32,f32,f32){
    (lerp(a.0,b.0,t), lerp(a.1,b.1,t), lerp(a.2,b.2,t))
}
fn ang_dist(a:Vec3, b:Vec3)->f32{ clamp01(0.5*(1.0 - a.dot(b))).acos() }

// ===== Shaders =====
fn shader_sun(n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let lat = (v-0.5)*PI;
    let lon = (u-0.5)*TAU;
    let s0 = 0.85 + 0.15*noise2(8.0*lon.cos(), 8.0*lat.sin());
    let gran = fbm2(12.0*u + t*0.2, 12.0*v - t*0.15, 5, 2.2, 0.5);
    let radial = 0.7 + 0.3*(1.0 - n.z*n.z);
    let flares = (0..4).map(|i|{
        let k = i as f32;
        let ang = t*0.2 + k*1.57;
        let dir = Vec3::new(ang.cos(), (ang*1.3).sin()*0.4, ang.sin()).norm();
        (1.0 - ang_dist(n, dir)/(0.25+0.1*(k%2.0))).max(0.0)
    }).fold(0.0, |a,b| a+b*0.15);
    let hot = s0*0.6 + 0.4*gran;
    let base = palette((0.95,0.5,0.1),(1.0,0.9,0.2), hot);
    (base.0*(radial+flares), base.1*(radial+flares*0.8), base.2*(radial*0.9+flares*0.6))
}
fn shader_rocky_cratered(n:Vec3, uv:(f32,f32), _t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let base = palette((0.35,0.32,0.3),(0.5,0.45,0.4), fbm2(4.0*u, 4.0*v, 4, 2.0, 0.5));
    let mut crater = 0.0;
    for i in 0..12 {
        let a = 0.3 * i as f32 + 1.37;
        let b = 0.5 * i as f32 + 0.73;
        let dir = Vec3::new(a.cos(), (a*1.7).sin()*0.6, b.sin()).norm();
        let d = ang_dist(n, dir);
        let r = 0.08 + 0.05*((a*3.1).sin()*0.5+0.5);
        let rim = smoothstep(r*1.05, r*0.9, d) - smoothstep(r*0.9, r*0.7, d);
        crater += rim*0.8 + smoothstep(r*0.6, r*0.4, d)*0.15;
    }
    (base.0*(1.0+crater*0.3), base.1*(1.0+crater*0.2), base.2*(1.0+crater*0.1))
}
fn shader_rocky_desert(_n:Vec3, uv:(f32,f32), _t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let dunes = ( ( (u*TAU*12.0).sin() * 0.5 + 0.5 ) * 0.6 + fbm2(u*10.0, v*10.0, 4, 2.2, 0.5)*0.4 ).min(1.0);
    palette((0.62,0.46,0.28),(0.9,0.7,0.45), dunes)
}
fn shader_ocean_world(_n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let continents = fbm2(u*3.0 + 0.1*t, v*3.0 - 0.07*t, 6, 2.0, 0.5);
    let shore = smoothstep(0.48, 0.52, continents);
    let land = palette((0.08,0.25,0.05),(0.3,0.55,0.22), fbm2(u*10.0, v*10.0, 3, 2.1, 0.5));
    let sand = (0.85,0.8,0.55);
    let water = palette((0.03,0.15,0.3),(0.05,0.35,0.7), 0.6+0.4*fbm2(u*8.0, v*8.0, 4, 2.3, 0.5));
    let clouds = smoothstep(0.6, 0.75, fbm2(u*7.0 + 0.1*t, v*12.0 + 0.07*t, 5, 2.1, 0.5));
    let base = if continents > 0.5 {
        let beach = shore - smoothstep(0.5, 0.52, continents);
        (lerp(land.0, sand.0, beach), lerp(land.1, sand.1, beach), lerp(land.2, sand.2, beach))
    } else { water };
    (lerp(base.0, 1.0, clouds*0.6), lerp(base.1, 1.0, clouds*0.6), lerp(base.2, 1.0, clouds*0.6))
}
fn shader_volcanic(_n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let basalt = palette((0.05,0.05,0.05),(0.15,0.12,0.1), fbm2(u*6.0, v*6.0, 4, 2.0, 0.5));
    let lava_flow = smoothstep(0.6, 0.8, fbm2(u*18.0 + t*0.2, v*18.0 - t*0.25, 5, 2.2, 0.55));
    let lava = palette((0.8,0.1,0.0),(1.0,0.7,0.2), fbm2(u*8.0, v*8.0, 4, 2.2, 0.5));
    (lerp(basalt.0, lava.0, lava_flow*0.7), lerp(basalt.1, lava.1, lava_flow*0.7), lerp(basalt.2, lava.2, lava_flow*0.7))
}
fn shader_gas_bands(_n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let lat = (v-0.5)*PI;
    let bands = (lat*14.0 + fbm2(u*8.0, v*2.0, 3, 2.0, 0.5)*2.0 - t*0.4).sin()*0.5+0.5;
    let base = palette((0.7,0.6,0.45),(0.95,0.85,0.7), bands);
    let spot_c = (0.65, 0.35, 0.2);
    let ang = ( (u-0.25 - 0.02*t)*TAU ).cos()*((v-0.55)*PI).cos();
    let spot = smoothstep(0.985, 0.995, ang);
    (lerp(base.0, spot_c.0, spot), lerp(base.1, spot_c.1, spot), lerp(base.2, spot_c.2, spot))
}
fn shader_gas_rings(_n:Vec3, uv:(f32,f32), _t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let lat = (v-0.5)*PI;
    let bands = (lat*10.0).sin()*0.5 + 0.5;
    let broken = bands*0.7 + fbm2(u*20.0, v*10.0, 4, 2.0, 0.5)*0.3;
    palette((0.85,0.75,0.6),(0.95,0.9,0.8), broken)
}
fn shader_ice_soft(_n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let lat = (v-0.5)*PI;
    let bands = (lat*4.0 - t*0.2).sin()*0.5+0.5;
    palette((0.6,0.8,0.85),(0.7,0.9,0.95), bands*0.8 + fbm2(u*5.0, v*5.0, 3, 2.0, 0.5)*0.2)
}
fn shader_ice_turbulent(_n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    let (u,v) = uv;
    let flow = fbm2(u*18.0 + t*0.1, v*18.0 - t*0.15, 5, 2.0, 0.5);
    palette((0.1,0.2,0.55),(0.15,0.35,0.9), flow)
}

fn ring_color(rr:f32) -> (f32,f32,f32,f32){
    let base = palette((0.8,0.75,0.65),(0.95,0.9,0.85), smoothstep(0.0,1.0,rr));
    let fine = ( (rr*200.0).sin()*0.5 + 0.5 )*0.25;
    let med  = ( (rr*30.0).sin()*0.5 + 0.5 )*0.35;
    let gap  = (rr*6.0 + 0.3).sin()*0.5 + 0.5;
    let a = (0.65 + 0.35*gap) * (0.6 + 0.4*noise2(rr*10.0, rr*10.0));
    (base.0+fine+med*0.2, base.1+fine*0.7+med*0.15, base.2+fine*0.4+med*0.1, a.min(0.95))
}

fn shade(body:&Body, n:Vec3, uv:(f32,f32), t:f32)->(f32,f32,f32){
    match body {
        Body::Sun => shader_sun(n,uv,t),
        Body::RockyCratered => shader_rocky_cratered(n,uv,t),
        Body::RockyDesert => shader_rocky_desert(n,uv,t),
        Body::OceanWorld => shader_ocean_world(n,uv,t),
        Body::Volcanic => shader_volcanic(n,uv,t),
        Body::GasBands => shader_gas_bands(n,uv,t),
        Body::GasRings => shader_gas_rings(n,uv,t),
        Body::IceSoft => shader_ice_soft(n,uv,t),
        Body::IceTurbulent => shader_ice_turbulent(n,uv,t),
        _ => (1.0,0.0,1.0)
    }
}

// Render con luz puntual (Sol), sombras y cámara
fn render_scene_rgb8(buf:&mut [u32], w:u32, h:u32, scene:&Scene, t:f32, samples:u32, cam:&Camera, light_intensity:f32, u_phase:f32){
    let aspect = w as f32 / h as f32;
    let (f,r,u,scale) = cam.basis(aspect);

    // Buscar Sol
    let sun_idx = scene.spheres.iter().position(|s| matches!(s.body, Body::Sun));
    let sun_c = sun_idx.map(|i| scene.spheres[i].c).unwrap_or(Vec3::new(-5.0,0.0,-5.0));
    let sun_r = sun_idx.map(|i| scene.spheres[i].r).unwrap_or(1.0);

    let u_phase_turns = u_phase / TAU; // fase adicional en vueltas

    buf.par_chunks_mut(1).enumerate().for_each(|(idx, px)|{
        let x = (idx as u32) % w;
        let y = (idx as u32) / w;
        let mut col = (0.0, 0.0, 0.0);
        let spp = samples as i32;
        for sy in 0..spp {
            for sx in 0..spp {
                let uu = ( (x as f32 + (sx as f32 + 0.5)/spp as f32) / w as f32 )*2.0 - 1.0;
                let vv = ( (y as f32 + (sy as f32 + 0.5)/spp as f32) / h as f32 )*2.0 - 1.0;
                let dir = f.add(r.mul(uu*aspect*scale)).add(u.mul(-vv*scale)).norm();
                let ray = Ray{ o:cam.pos, d:dir };
                let mut best_t = f32::INFINITY;
                let mut best_i = usize::MAX;
                let mut best_hit : Option<Hit> = None;
                for (i, s) in scene.spheres.iter().enumerate() {
                    if let Some(mut hit) = intersect_sphere(&ray, s) {
                        if hit.t < best_t {
                            let (mut u1, v1) = hit.uv;
                            // Rotación: auto-spin (según tiempo) + fase manual (flechas)
                            u1 = fract(u1 + (s.rot_speed * t) / TAU + u_phase_turns);
                            hit.uv = (u1, v1);
                            hit.id = i;
                            best_t = hit.t;
                            best_i = i;
                            best_hit = Some(hit);
                        }
                    }
                }
                let mut ring_col = None::<(f32,f32,f32,f32,f32)>;
                for r1 in scene.rings.iter() {
                    if let Some((t_hit, dist)) = intersect_ring(&ray, r1) {
                        if t_hit < best_t {
                            let rr = (dist - r1.r_inner) / (r1.r_outer - r1.r_inner);
                            let (cr,cg,cb,ca) = ring_color(rr);
                            ring_col = Some((cr,cg,cb,ca,t_hit));
                            best_t = t_hit - 1e-5;
                        }
                    }
                }
                let mut c = (0.0,0.0,0.0);
                if let Some(hit) = best_hit {
                    let sref = &scene.spheres[best_i];
                    let p = ray.o.add(ray.d.mul(hit.t));
                    // Luz puntual
                    let ldir = sun_c.sub(p).norm();
                    let ldist = sun_c.sub(p).len() - sun_r;
                    let mut in_shadow = false;
                    if let Some(_) = sun_idx {
                        let shadow_ray = Ray{ o: p.add(ldir.mul(1e-3)), d: ldir };
                        for (j, sph) in scene.spheres.iter().enumerate() {
                            if j == best_i { continue; }
                            if let Some(h2) = intersect_sphere(&shadow_ray, sph) {
                                if h2.t < ldist { in_shadow = true; break; }
                            }
                        }
                    }
                    let base = shade(&sref.body, hit.n, hit.uv, t);
                    let ndl = hit.n.dot(ldir).max(0.0);
                    let amb = AMBIENT_BASE;
                    let diff = if in_shadow { 0.0 } else { ndl * light_intensity };
                    let rim = (1.0 - hit.n.dot(ray.d).abs()).powf(2.0)*RIM_INTENSITY;
                    c = (base.0*(amb + diff) + rim, base.1*(amb + diff) + rim, base.2*(amb + diff) + rim);
                } else {
                    let star = smoothstep(0.995, 1.0, noise2((x as f32)*0.73, (y as f32)*0.51));
                    c = (star, star, star);
                }
                if let Some((rr,rg,rb,ra,_t)) = ring_col {
                    c = ( lerp(c.0, rr, ra), lerp(c.1, rg, ra), lerp(c.2, rb, ra) );
                }
                col.0 += c.0; col.1 += c.1; col.2 += c.2;
            }
        }
        let inv = 1.0 / (spp*spp) as f32;
        let r8 = (clamp01(col.0*inv)*255.0) as u32;
        let g8 = (clamp01(col.1*inv)*255.0) as u32;
        let b8 = (clamp01(col.2*inv)*255.0) as u32;
        px[0] = (r8 << 16) | (g8 << 8) | b8;
    });
}

fn build_scene_system(t:f32, animate_orbits:bool) -> Scene {
    let sun = Sphere{ c:Vec3::new(-3.5, 0.0, -3.0), r:1.2, body:Body::Sun, rot_speed:0.15 };
    let mut spheres = vec![sun];
    let layout = [
        (Body::RockyCratered, 0.9, 0.35, 0.8),
        (Body::RockyDesert,   1.4, 0.40, 0.5),
        (Body::OceanWorld,    2.1, 0.48, 0.3),
        (Body::Volcanic,      2.9, 0.42, 0.25),
        (Body::GasBands,      3.9, 0.80, 0.18),
        (Body::GasRings,      5.2, 0.75, 0.12),
        (Body::IceSoft,       6.4, 0.55, 0.10),
        (Body::IceTurbulent,  7.6, 0.55, 0.08),
    ];
    for (i,(b, orbit, r, rot)) in layout.iter().enumerate() {
        let phase = if animate_orbits { t * 0.25 / (*orbit) } else { 0.0 };
        let cx = -0.5 + (orbit * ORBIT_SCALE * phase.cos());
        let cz = -2.4 + (orbit * ORBIT_SCALE * phase.sin());
        let cy = if i % 2 == 0 { 0.1 } else { -0.1 };
        spheres.push(Sphere{ c:Vec3::new(cx as f32, cy as f32, cz as f32), r:*r, body:*b, rot_speed:*rot });
    }

    // Luna para el planeta rocoso (primer RockyCratered en la lista)
    if ROCKY_MOON_ENABLED_SYSTEM {
        if let Some((idx, host)) = spheres.iter().enumerate().find(|(_i,s)| matches!(s.body, Body::RockyCratered)) {
            let mp = t * ROCKY_MOON_SPEED;
            spheres.push(Sphere{
                c: host.c.add(Vec3::new(ROCKY_MOON_DIST*host.r * mp.cos(), ROCKY_MOON_TILT*host.r, ROCKY_MOON_DIST*host.r * mp.sin())),
                r: ROCKY_MOON_SCALE * host.r,
                body: Body::RockyCratered,
                rot_speed: 0.3
            });
        }
    }

    // Anillos para GasRings
    let ring_host_idx = 6; // coincide con GasRings en layout
    let ring_host = spheres[ring_host_idx];
    let rings = vec![ Ring{
        c: ring_host.c,
        n: Vec3::new(0.0, 1.0, 0.25).norm(),
        r_inner: ring_host.r*1.2,
        r_outer: ring_host.r*2.0,
    } ];
    Scene{ spheres, rings }
}

fn build_single(body:Body, t:f32) -> Scene {
    let c = Vec3::new(0.0, 0.0, -2.5);
    let r = match body {
        Body::Sun => 1.1,
        Body::GasBands|Body::GasRings => 0.95,
        _ => 0.7
    };
    let mut spheres = vec![ Sphere{ c, r, body, rot_speed:0.5 } ];
    // Anillos si aplica
    let mut rings = vec![];
    if matches!(body, Body::GasRings) {
        rings.push(Ring{ c, n:Vec3::new(0.0, 1.0, 0.2).norm(), r_inner: r*1.2, r_outer: r*2.0 });
    }
    // Luna para el planeta rocoso en vista individual
    if ROCKY_MOON_ENABLED_SINGLE && matches!(body, Body::RockyCratered) {
        let mp = t * ROCKY_MOON_SPEED;
        spheres.push(Sphere{
            c: c.add(Vec3::new(ROCKY_MOON_DIST*r * mp.cos(), ROCKY_MOON_TILT*r, ROCKY_MOON_DIST*r * mp.sin())),
            r: ROCKY_MOON_SCALE * r,
            body: Body::RockyCratered,
            rot_speed: 0.25
        });
    }
    Scene{ spheres, rings }
}

fn ensure_out_dir(dir:&std::path::Path){ let _ = std::fs::create_dir_all(dir); }

fn main(){
    let opts = Opts::parse();

    // ----- Ventana interactiva -----
    if opts.window {
        let title = if opts.gallery_top6 || !matches!(opts.body, Body::System) {
            "proc_solar — Vista individual | ←/→: rotar objeto | ↑/↓: dolly | [ ]: cambiar cuerpo | SPACE órbita/libre | ESC salir"
        } else {
            "proc_solar — Sistema | Mouse+teclas | ESC salir"
        };
        let mut window = Window::new(
            title,
            opts.width as usize,
            opts.height as usize,
            WindowOptions::default(),
        ).expect("Failed to create window");
        window.limit_update_rate(Some(std::time::Duration::from_micros(1_000_000 / opts.fps as u64)));

        let mut buf = vec![0u32; (opts.width*opts.height) as usize];
        let mut cam = Camera{ pos: Vec3::new(0.0, 0.0, 5.5), yaw: 0.0, pitch: 0.0, fov_deg: 50.0 };
        let mut orbit = Orbit{ enabled: ORBIT_MODE_DEFAULT, target: Vec3::new(-3.5, 0.0, -3.0), distance: 6.0 };
        let mut prev = Instant::now();
        let start = Instant::now();

        // Estado de galería / individual
        let mut gallery_idx: usize = 0;
        let mut current_body = if opts.gallery_top6 {
            TOP6[gallery_idx]
        } else if matches!(opts.body, Body::System) {
            Body::System
        } else {
            opts.body
        };

        // Fase manual de rotación del objeto (aplicado a las coordenadas U del shader)
        let mut obj_phase: f32 = 0.0; // rad

        // Estado del mouse
        let mut last_mouse: Option<(f32,f32)> = None;

        while window.is_open() {
            if window.is_key_down(Key::Escape) { break; }

            let now = Instant::now();
            let dt = (now - prev).as_secs_f32();
            prev = now;
            let t = start.elapsed().as_secs_f32();

            // Cambiar cuerpo con [ ]
            if window.is_key_pressed(Key::LeftBracket, KeyRepeat::No) && opts.gallery_top6 {
                gallery_idx = (gallery_idx + TOP6.len() - 1) % TOP6.len();
                current_body = TOP6[gallery_idx];
                obj_phase = 0.0;
            }
            if window.is_key_pressed(Key::RightBracket, KeyRepeat::No) && opts.gallery_top6 {
                gallery_idx = (gallery_idx + 1) % TOP6.len();
                current_body = TOP6[gallery_idx];
                obj_phase = 0.0;
            }

            // SPACE: alterna órbita/libre
            if window.is_key_pressed(Key::Space, KeyRepeat::No) {
                orbit.enabled = !orbit.enabled;
            }

            // Pitch y FOV por teclado
            let pitch_speed = 1.0;
            if window.is_key_down(Key::PageUp)   { cam.pitch = (cam.pitch + pitch_speed*dt).clamp(-1.2, 1.2); }
            if window.is_key_down(Key::PageDown) { cam.pitch = (cam.pitch - pitch_speed*dt).clamp(-1.2, 1.2); }
            if window.is_key_down(Key::Equal) { cam.fov_deg = (cam.fov_deg - 40.0*dt).clamp(20.0, 90.0); }
            if window.is_key_down(Key::Minus) { cam.fov_deg = (cam.fov_deg + 40.0*dt).clamp(20.0, 90.0); }

            // Rueda mouse: FOV
            if let Some((_sx, sy)) = window.get_scroll_wheel() {
                cam.fov_deg = (cam.fov_deg - sy * 2.5).clamp(20.0, 90.0);
            }

            // Vista individual redefine flechas
            let single_view = opts.gallery_top6 || !matches!(opts.body, Body::System);
            if single_view {
                // Auto-spin constante
                obj_phase += AUTO_SPIN_BASE * dt;
                // ←/→: rotar orientación manualmente
                if window.is_key_down(Key::Left)  { obj_phase -= 1.2 * dt; }
                if window.is_key_down(Key::Right) { obj_phase += 1.2 * dt; }
                // ↑/↓: dolly
                let (f,_,_,_) = cam.basis(opts.width as f32 / opts.height as f32);
                if window.is_key_down(Key::Up)   { cam.pos = cam.pos.add(f.mul(3.0*dt)); }
                if window.is_key_down(Key::Down) { cam.pos = cam.pos.add(f.mul(-3.0*dt)); }
            } else {
                // En sistema, flechas controlan yaw
                let yaw_speed = 1.2;
                if window.is_key_down(Key::Left)  { cam.yaw -= yaw_speed * dt; }
                if window.is_key_down(Key::Right) { cam.yaw += yaw_speed * dt; }
            }

            // Movimiento WASD/QE
            let (f,r,u,_) = cam.basis(opts.width as f32 / opts.height as f32);
            if window.is_key_down(Key::A) { cam.pos = cam.pos.add(r.mul(-3.0*dt)); if orbit.enabled { orbit.target = orbit.target.add(r.mul(-3.0*dt)); } }
            if window.is_key_down(Key::D) { cam.pos = cam.pos.add(r.mul( 3.0*dt)); if orbit.enabled { orbit.target = orbit.target.add(r.mul( 3.0*dt)); } }
            if window.is_key_down(Key::W) { cam.pos = cam.pos.add(f.mul( 3.0*dt)); if orbit.enabled { orbit.target = orbit.target.add(f.mul( 3.0*dt)); } }
            if window.is_key_down(Key::S) { cam.pos = cam.pos.add(f.mul(-3.0*dt)); if orbit.enabled { orbit.target = orbit.target.add(f.mul(-3.0*dt)); } }
            if window.is_key_down(Key::Q) { cam.pos = cam.pos.add(u.mul(-3.0*dt)); if orbit.enabled { orbit.target = orbit.target.add(u.mul(-3.0*dt)); } }
            if window.is_key_down(Key::E) { cam.pos = cam.pos.add(u.mul( 3.0*dt)); if orbit.enabled { orbit.target = orbit.target.add(u.mul( 3.0*dt)); } }

            // Mouse
            if let Some((mx,my)) = window.get_mouse_pos(MouseMode::Clamp) {
                if let Some((pmx,pmy)) = last_mouse {
                    let dx = mx - pmx;
                    let dy = my - pmy;
                    if orbit.enabled {
                        if window.get_mouse_down(MouseButton::Left) {
                            cam.yaw   += dx * 0.005;
                            cam.pitch -= dy * 0.005;
                            cam.pitch = cam.pitch.clamp(-1.2, 1.2);
                        }
                        let fwd = cam.forward();
                        let world_up = Vec3::new(0.0,1.0,0.0);
                        let right = world_up.cross(fwd).norm();
                        let up = fwd.cross(right).norm();
                        if window.get_mouse_down(MouseButton::Middle) {
                            orbit.target = orbit.target.add( right.mul(-dx * 0.005) );
                            orbit.target = orbit.target.add( up.mul( dy * 0.005) );
                        }
                        if window.get_mouse_down(MouseButton::Right) {
                            orbit.distance = (orbit.distance - dy * 0.01).clamp(0.8, 60.0);
                        }
                        let fwd2 = cam.forward();
                        cam.pos = orbit.target.sub( fwd2.mul(orbit.distance) );
                    } else {
                        if window.get_mouse_down(MouseButton::Left) {
                            cam.yaw   += dx * 0.005;
                            cam.pitch -= dy * 0.005;
                            cam.pitch = cam.pitch.clamp(-1.2, 1.2);
                        }
                        let (f2,r2,u2,_) = cam.basis(opts.width as f32 / opts.height as f32);
                        if window.get_mouse_down(MouseButton::Middle) {
                            cam.pos = cam.pos.add( r2.mul(-dx * 0.005) );
                            cam.pos = cam.pos.add( u2.mul( dy * 0.005) );
                        }
                        if window.get_mouse_down(MouseButton::Right) {
                            cam.pos = cam.pos.add( f2.mul(-dy * 0.01) );
                        }
                    }
                }
                last_mouse = Some((mx,my));
            } else {
                last_mouse = None;
            }

            // Render
            let single_view = opts.gallery_top6 || !matches!(opts.body, Body::System);
            if single_view {
                let scene = build_single(current_body, t);
                render_scene_rgb8(&mut buf, opts.width, opts.height, &scene, t, 1, &cam, SUN_LIGHT_INTENSITY, obj_phase);
            } else {
                let scene = build_scene_system(t, opts.orbits);
                render_scene_rgb8(&mut buf, opts.width, opts.height, &scene, t, 1, &cam, SUN_LIGHT_INTENSITY, 0.0);
            }
            window.update_with_buffer(&buf, opts.width as usize, opts.height as usize).unwrap();
        }
        return;
    }

    // ----- PNG offline -----
    ensure_out_dir(&opts.out);
    let samples = if opts.hq { 3 } else { 1 };
    let bodies: Vec<Body> = match opts.body {
        Body::All => vec![
            Body::Sun, Body::RockyCratered, Body::RockyDesert, Body::OceanWorld,
            Body::Volcanic, Body::GasBands, Body::GasRings, Body::IceSoft, Body::IceTurbulent
        ],
        Body::System => vec![Body::System],
        b => vec![b]
    };

    for b in bodies {
        let (scene, name) = if matches!(b, Body::System) {
            (build_scene_system(opts.time, false), "system".to_string())
        } else {
            (build_single(b, opts.time), format!("{:?}", b).to_lowercase())
        };
        let mut img = RgbImage::new(opts.width, opts.height);
        let mut buf = vec![0u32; (opts.width*opts.height) as usize];
        let cam = Camera{ pos: Vec3::new(0.0, 0.0, 5.0), yaw: 0.0, pitch: 0.0, fov_deg: 50.0 };
        render_scene_rgb8(&mut buf, opts.width, opts.height, &scene, opts.time, samples, &cam, SUN_LIGHT_INTENSITY, 0.0);
        for (i, pixel) in img.pixels_mut().enumerate() {
            let v = buf[i];
            let r = ((v >> 16) & 0xFF) as u8;
            let g = ((v >> 8) & 0xFF) as u8;
            let b = (v & 0xFF) as u8;
            *pixel = image::Rgb([r,g,b]);
        }
        let mut path = opts.out.clone();
        path.push(format!("{}.png", name));
        img.save(&path).expect("failed to write png");
        println!("Wrote {:?}", path);
    }
}
