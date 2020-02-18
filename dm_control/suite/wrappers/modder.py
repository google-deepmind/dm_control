
from collections import defaultdict
import numpy as np

class BaseModder():

    def __init__(self, physics, random_state=None):
        self.physics=physics
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            # random_state assumed to be an int
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    @property
    def model(self):
        # Available for quick convenience access
        return self.physics.named.model

class LightModder(BaseModder):

    def __init__(self, *args, **kwargs):
        super(LightModder,self).__init__(*args, **kwargs)

    def get_pos(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name



        return self.model.light_pos[lightid]

    def get_dir(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name


        return self.model.light_dir[lightid]

    def get_active(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_active[lightid]

    def get_specular(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name


        return self.model.light_specular[lightid]

    def get_ambient(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name


        return self.model.light_ambient[lightid]

    def get_diffuse(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name



        return self.model.light_diffuse[lightid]

    def get_castshadow(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name
        return self.model.light_castshadow[lightid]

    def set_pos(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_pos[lightid] = value

    def set_dir(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_dir[lightid] = value

    def set_active(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        self.model.light_active[lightid] = value

    def set_specular(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_specular[lightid] = value

    def set_ambient(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_ambient[lightid] = value

    def set_diffuse(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_diffuse[lightid] = value

    def set_castshadow(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name
        self.model.light_castshadow[lightid] = value

    def get_lightid(self, name):
        return self.model.light_bodyid[name]


class CameraModder(BaseModder):

    def __init__(self, *args, **kwargs):
        super(CameraModder,self).__init__(*args, **kwargs)

    def get_fovy(self,name):
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_fovy[camid]

    def set_fovy(self, name, value):
        camid = self.get_camid(name)
        assert 0 < value < 180
        assert camid > -1, "Unknown camera %s" % name
        self.model.cam_fovy[camid] = value

    def get_quat(self, name):
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_quat[camid]

    def set_quat(self, name, value):
        value = list(value)
        assert len(value) == 4, (
            "Expectd value of length 3, instead got %s" % value)
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        self.model.cam_quat[camid] = value

    def get_pos(self, name):
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_pos[camid]

    def set_pos(self, name, value):
        value = list(value)
        assert len(value) == 3, (
            "Expected value of length 3, instead got %s" % value)
        camid = self.get_camid(name)
        assert camid > -1
        self.model.cam_pos[camid] = value

    def get_camid(self, name):
        return self.model.cam_bodyid[name]


class MaterialModder(BaseModder):
    """
    Modify material properties of a model. Example use:
        sim = MjSim(...)
        modder = MaterialModder(sim)
        modder.set_specularity('some_geom', 0.5)
        modder.rand_all('another_geom')
    """

    def __init__(self, *args, **kwargs):
        super(MaterialModder,self).__init__(*args, **kwargs)

    def set_specularity(self, name, value):
        assert 0 <= value <= 1.0
        mat_id = self.get_mat_id(name)
        self.model.mat_specular[mat_id] = value

    def set_shininess(self, name, value):
        assert 0 <= value <= 1.0
        mat_id = self.get_mat_id(name)
        self.model.mat_shininess[mat_id] = value

    def set_reflectance(self, name, value):
        assert 0 <= value <= 1.0
        mat_id = self.get_mat_id(name)
        self.model.mat_reflectance[mat_id] = value

    def set_texrepeat(self, name, repeat_x, repeat_y):
        mat_id = self.get_mat_id(name)
        # ensure the following is set to false, so that repeats are
        # relative to the extent of the body.
        self.model.mat_texuniform[mat_id] = 0
        self.model.mat_texrepeat[mat_id, :] = [repeat_x, repeat_y]

    def rand_all(self, name):
        self.rand_specularity(name)
        self.rand_shininess(name)
        self.rand_reflectance(name)

    def rand_specularity(self, name):
        value = 0.1 + 0.2 * self.random_state.uniform()
        self.set_specularity(name, value)

    def rand_shininess(self, name):
        value = 0.1 + 0.5 * self.random_state.uniform()
        self.set_shininess(name, value)

    def rand_reflectance(self, name):
        value = 0.1 + 0.5 * self.random_state.uniform()
        self.set_reflectance(name, value)

    def rand_texrepeat(self, name, max_repeat=5):
        repeat_x = self.random_state.randint(0, max_repeat) + 1
        repeat_y = self.random_state.randint(0, max_repeat) + 1
        self.set_texrepeat(name, repeat_x, repeat_y)

    def get_mat_id(self, name):
        """ Returns the material id based on the geom name. """

        return self.model.geom_matid[name]


class TextureModder(BaseModder):
    """
    Modify textures in model. Example use:
        sim = MjSim(...)
        modder = TextureModder(sim)
        modder.whiten_materials()  # ensures materials won't impact colors
        modder.set_checker('some_geom', (255, 0, 0), (0, 0, 0))
        modder.rand_all('another_geom')
    Note: in order for the textures to take full effect, you'll need to set
    the rgba values for all materials to [1, 1, 1, 1], otherwise the texture
    colors will be modulated by the material colors. Call the
    `whiten_materials` helper method to set all material colors to white.
    """

    def __init__(self, *args, **kwargs):
        super(TextureModder,self).__init__(*args, **kwargs)

        self.textures = [Texture(self.model, i)
                         for i in range(self.model.ntex)]
        self._build_tex_geom_map()

        # These matrices will be used to rapidly synthesize
        # checker pattern bitmaps
        self._cache_checker_matrices()

    def get_texture(self, name):
        if name == 'skybox':
            tex_id = -1
            for i in range(self.model.ntex):
                # TODO: Don't hardcode this
                skybox_textype = 2
                if self.model.tex_type[i] == skybox_textype:
                    tex_id = i
            assert tex_id >= 0, "Model has no skybox"
        else:
            geom_id = self.model.geom_name2id(name)
            mat_id = self.model.geom_matid[geom_id]
            assert mat_id >= 0, "Geom has no assigned material"
            tex_id = self.model.mat_texid[mat_id]
            assert tex_id >= 0, "Material has no assigned texture"

        texture = self.textures[tex_id]

        return texture

    def get_checker_matrices(self, name):
        if name == 'skybox':
            return self._skybox_checker_mat
        else:
            geom_id = self.model.geom_name2id(name)
            return self._geom_checker_mats[geom_id]

    def set_checker(self, name, rgb1, rgb2):
        bitmap = self.get_texture(name).bitmap
        cbd1, cbd2 = self.get_checker_matrices(name)

        rgb1 = np.asarray(rgb1).reshape([1, 1, -1])
        rgb2 = np.asarray(rgb2).reshape([1, 1, -1])
        bitmap[:] = rgb1 * cbd1 + rgb2 * cbd2

        self.upload_texture(name)
        return bitmap

    def set_gradient(self, name, rgb1, rgb2, vertical=True):
        """
        Creates a linear gradient from rgb1 to rgb2.
        Args:
        - rgb1 (array): start color
        - rgb2 (array): end color
        - vertical (bool): if True, the gradient in the positive
            y-direction, if False it's in the positive x-direction.
        """
        # NOTE: MuJoCo's gradient uses a sigmoid. Here we simplify
        # and just use a linear gradient... We could change this
        # to just use a tanh-sigmoid if needed.
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        if vertical:
            p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
        else:
            p = np.tile(np.linspace(0, 1, w), (h, 1))

        for i in range(3):
            bitmap[..., i] = rgb2[i] * p + rgb1[i] * (1.0 - p)

        self.upload_texture(name)
        return bitmap

    def set_rgb(self, name, rgb):
        bitmap = self.get_texture(name).bitmap
        bitmap[..., :] = np.asarray(rgb)

        self.upload_texture(name)
        return bitmap

    def set_noise(self, name, rgb1, rgb2, fraction=0.9):
        """
        Args:
        - name (str): name of geom
        - rgb1 (array): background color
        - rgb2 (array): color of random noise foreground color
        - fraction (float): fraction of pixels with foreground color
        """
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        mask = self.random_state.uniform(size=(h, w)) < fraction

        bitmap[..., :] = np.asarray(rgb1)
        bitmap[mask, :] = np.asarray(rgb2)

        self.upload_texture(name)
        return bitmap

    def randomize(self):
        for name in self.sim.model.geom_names:
            self.rand_all(name)

    def rand_all(self, name):
        choices = [
            self.rand_checker,
            self.rand_gradient,
            self.rand_rgb,
            self.rand_noise,
        ]
        choice = self.random_state.randint(len(choices))
        return choices[choice](name)

    def rand_checker(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        return self.set_checker(name, rgb1, rgb2)

    def rand_gradient(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        vertical = bool(self.random_state.uniform() > 0.5)
        return self.set_gradient(name, rgb1, rgb2, vertical=vertical)

    def rand_rgb(self, name):
        rgb = self.get_rand_rgb()
        return self.set_rgb(name, rgb)

    def rand_noise(self, name):
        fraction = 0.1 + self.random_state.uniform() * 0.8
        rgb1, rgb2 = self.get_rand_rgb(2)
        return self.set_noise(name, rgb1, rgb2, fraction)

    def upload_texture(self, name):
        """
        Uploads the texture to the GPU so it's available in the rendering.
        """
        texture = self.get_texture(name)
        if not self.sim.render_contexts:
            cymj.MjRenderContextOffscreen(self.sim)
        for render_context in self.sim.render_contexts:
            render_context.upload_texture(texture.id)

    def whiten_materials(self, geom_names=None):
        """
        Helper method for setting all material colors to white, otherwise
        the texture modifications won't take full effect.
        Args:
        - geom_names (list): list of geom names whose materials should be
            set to white. If omitted, all materials will be changed.
        """
        geom_names = geom_names or []
        if geom_names:
            for name in geom_names:
                geom_id = self.model.geom_name2id(name)
                mat_id = self.model.geom_matid[geom_id]
                self.model.mat_rgba[mat_id, :] = 1.0
        else:
            self.model.mat_rgba[:] = 1.0

    def get_rand_rgb(self, n=1):
        def _rand_rgb():
            return np.array(self.random_state.uniform(size=3) * 255,
                            dtype=np.uint8)

        if n == 1:
            return _rand_rgb()
        else:
            return tuple(_rand_rgb() for _ in range(n))

    def _build_tex_geom_map(self):
        # Build a map from tex_id to geom_ids, so we can check
        # for collisions.
        self._geom_ids_by_tex_id = defaultdict(list)
        for geom_id in range(self.model.ngeom):
            mat_id = self.model.geom_matid[geom_id]
            if mat_id >= 0:
                tex_id = self.model.mat_texid[mat_id]
                if tex_id >= 0:
                    self._geom_ids_by_tex_id[tex_id].append(geom_id)

    def _cache_checker_matrices(self):
        """
        Cache two matrices of the form [[1, 0, 1, ...],
                                        [0, 1, 0, ...],
                                        ...]
        and                            [[0, 1, 0, ...],
                                        [1, 0, 1, ...],
                                        ...]
        for each texture. To use for fast creation of checkerboard patterns
        """
        self._geom_checker_mats = []
        for geom_id in range(self.model.ngeom):
            mat_id = self.model.geom_matid[geom_id]
            tex_id = self.model.mat_texid[mat_id]
            texture = self.textures[tex_id]
            h, w = texture.bitmap.shape[:2]
            self._geom_checker_mats.append(self._make_checker_matrices(h, w))

        # add skybox
        skybox_tex_id = -1
        for tex_id in range(self.model.ntex):
            skybox_textype = 2
            if self.model.tex_type[tex_id] == skybox_textype:
                skybox_tex_id = tex_id
        if skybox_tex_id >= 0:
            texture = self.textures[skybox_tex_id]
            h, w = texture.bitmap.shape[:2]
            self._skybox_checker_mat = self._make_checker_matrices(h, w)
        else:
            self._skybox_checker_mat = None

    def _make_checker_matrices(self, h, w):
        re = np.r_[((w + 1) // 2) * [0, 1]]
        ro = np.r_[((w + 1) // 2) * [1, 0]]
        cbd1 = np.expand_dims(np.row_stack(((h + 1) // 2) * [re, ro]), -1)[:h, :w]
        cbd2 = np.expand_dims(np.row_stack(((h + 1) // 2) * [ro, re]), -1)[:h, :w]
        return cbd1, cbd2




MJT_TEXTURE_ENUM = ['texcorner','texcorner4','grid','texplane','texcolormap' 'background', 'skybox',]

class Texture():
    """
    Helper class for operating on the MuJoCo textures.
    """

    __slots__ = ['id', 'type', 'height', 'width', 'tex_adr', 'tex_rgb']

    def __init__(self, model, tex_id):
        self.id = tex_id
        self.type = MJT_TEXTURE_ENUM[model.tex_type[tex_id]]
        self.height = model.tex_height[tex_id]
        self.width = model.tex_width[tex_id]
        self.tex_adr = model.tex_adr[tex_id]
        self.tex_rgb = model.tex_rgb

    @property
    def bitmap(self):
        size = self.height * self.width * 3
        data = self.tex_rgb[self.tex_adr:self.tex_adr + size]
        return data.reshape((self.height, self.width, 3))
