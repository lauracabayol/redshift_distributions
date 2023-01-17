import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class corr2pt_estimator:
    def __init__(self, cat_path, ra_min, ra_max, dec_min, dec_max, mag_cut, min_sep= 0.03, max_sep = 26, nedges=8, njk = 25,nangbins = 8, verbosity=2):
        self.verbosity = verbosity
        self.min_sep= min_sep
        self.max_sep = max_sep
        self.nedges=nedges
        self.njk = njk
        self.nangbins = 
        
        dfin = pd.read_parquet('/data/astro/scratch/lcabayol/z_probdist/FS2_1sqdeg.parquet')
        dfin['imag'] = -2.5*np.log10(dfin.subaru_i_el_model3_odonnell_ext) - 48.6
        dfin = dfin[dfin.imag <mag_cut]
        dfin = dfin[(dfin.ra_gal >ra_min)&(dfin.ra_gal <ra_max)&(dfin.dec_gal >dec_min)&(dfin.dec_gal <dec_max)]
        
        
        
        if self.verbosity >1:
            print('The catalogue contains %s objects within a ra,dec range (%s,%s), (%s,%s)'%(len(dfin),ra_min, ra_max, dec_min, dec_max))
            dfin_ = dfin.sample(10000)
            plt.scatter(dfin_.ra_gal, dfin_dec_gal, s=2)
            plt.xlabel('RA')
            plt.ylabel('Dec')
            plt.show()
        
        
        ra_bins = np.linspace(ra_min,ra_max,np.sqrt(njk)+1)
        dfin['jk_ra'] = pd.cut(dfin['ra_gal'].values, ra_bins,labels=np.arange(0,5,1))

        dec_bins = np.linspace(dec_min,dec_max,np.sqrt(njk)+1)
        dfin['jk_dec'] = pd.cut(dfin['dec_gal'].values, dec_bins,labels=np.arange(0,5,1))

        self.cat = dfin
        
        if self.verbosity >1:
            print('We will used %s jacknife regions '%njk)
            c = -1
            centers_fs = np.zeros(shape = (25,2))

            for i in range(1,int(np.sqrt(njk)+1)):
                for j in range(1,int(np.sqrt(njk)+1)):
                    c = c+1
                    df_jk = dfin[(dfin.jk_ra==i)&(dfin.jk_dec==j)]
                    cent_ra = 0.5*(df_jk.ra_gal.max()+df_jk.ra_gal.min())
                    cent_dec = 0.5*(df_jk.dec_gal.max()+df_jk.dec_gal.min())
                    centers_fs[c] = np.c_[cent_ra,cent_dec]

                    plt.scatter(df_jk.ra_gal,df_jk.dec_gal, color = color_palette[c] )

                    plt.scatter(cent_ra,cent_dec, color ='crimson')

                    plt.xlabel('RA', fontsize = 18)
                    plt.ylabel('Dec', fontsize = 18)

                    plt.xticks(fontsize = 12)
                    plt.yticks(fontsize = 12)


                    plt.title('Flagship', fontsize = 14)
                    
                    plt.show()
                    
            self.max_sep = self.max_sep * units_to_degrees('arcmin')
            self.min_sep = self.min_sep * units_to_degrees('arcmin')
            
            th = np.linspace(np.log10(self.min_sep), np.log10(self.max_sep), self.nedges)
            thetac = 10**np.array([(th[i]+th[i+1])/2 for i in range(len(th)-1)])
            self.theta = 10**th 
            
            self.jk_centers = self._jacknife_centers()
            #Finds the minimum distance to a center for each center
            dist = np.array([np.sort([self._dist_cent(self.jk_centers[i],self.jk_centers[j]) for i in range(len(self.jk_centers))])[1] for j in range(len(self.jk_centers))])
            #Fixes double of this distance as the criteria for not considering correlations. 
            dist = dist*2.
            #This distance has to be at least 4 times the maximum angular separation considered.
            self.center_min_dis = np.array( [ 4.*max_sep if x < 4.*max_sep else x for x in dist] )
            
            a=np.concatenate((np.array([[(i,j) for i in range(self.njk)] for j in range(self.njk)])))
            sel = np.array([self._dist_cent(self.jk_centers[i],self.jk_centers[j]) < max(self.center_min_dis[i], self.center_min_dis[j]) for (i,j) in a])
            self.bins_to_calculate = a[sel]
            
            self.cat['jk_index'] = self.cat.jk_ra.values.astype(np.float) *5 + self.cat.jk_dec.values.astype(np.float)
            
        
        
    def _units_to_degrees(self, un):
        if un == 'arcmin':
            todeg = 1./60.
        elif un == 'arcsec':
            todeg = 1./60.**2.
        else:
            todeg = 1.

        return todeg

    def _thetaphiTOradec(self,theta,phi):
        dec = (np.pi/2. - theta)*(180./np.pi)
        ra = phi * (180./np.pi)
        return ra,dec

    def _radecTOthetaphi(ra,dec):
        theta = np.pi/2. - dec*(np.pi/180.)
        phi = ra * (np.pi/180.)
        return theta,phi
    
    def _dist_cent(self,a, b):
        """Angular distance between two centers (units: degrees). Makes use of spherical law of cosines.
        """
        todeg = np.pi/180.
        a = a*todeg
        b = b*todeg
        cos = np.sin(a[1])*np.sin(b[1]) + np.cos(a[1])*np.cos(b[1])*np.cos(a[0]-b[0])
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos)/todeg

    def _NN(self,ind):
        lengths = [len(ind[ind==n]) for n in range(njk)]
        return lengths
    
    def _compute_tree_distances(self, treeA, treeB, max_sep, theta ):

        tprs = treeA.sparse_distance_matrix(treeB,
                                max_distance=max_sep,
                                output_type='ndarray')

        pairs = np.array([((tprs['v']>theta[i])&(tprs['v']<theta[i+1])).sum() for i in range(len(theta)-1)]).astype(float)
        return pairs

    def _estimate(self, x):
        l = len(x)-1
        mean = np.mean(x)
        std = np.sqrt(l*np.mean(abs(x - mean)**2))
        return mean, std
    
    def _jacknife_centers(self):
        c = -1
        centers = np.zeros(shape = (25,2))
        for i in range(1,int(np.sqrt(njk)+1)):
            for j in range(1,int(np.sqrt(njk)+1)):        
                c = c+1
                df_jk = self.cat[(cat.jk_ra==i)&(cat.jk_dec==j)]
                cent_ra = 0.5*(df_jk.ra_rand.max()+df_jk.ra_rand.min())
                cent_dec = 0.5*(df_jk.dec_rand.max()+df_jk.dec_rand.min())

                centers[c] = np.c_[cent_ra,cent_dec]
                
        return centers
    
    def _create_randcat(self,nrand):
        ra_rands = np.random.uniform(self.cat.ra_gal.min(),self.cat.ra_gal.max(), nrand)
        dec_rands = (180/np.pi)*np.arcsin(np.random.uniform(np.sin(self.cat.dec_gal.min()*np.pi/180), np.sin(self.cat.dec_gal.max()*np.pi/180), nrand))
        
        df_rand = pd.DataFrame(np.c_[ra_rands,dec_rands], columns = ['ra_rand','dec_rand'])
        df_rand['jk_ra'] = pd.cut(df_rand['ra_rand'].values, np.linspace(self.cat.ra_gal.min(),self.cat.ra_gal.max(),int(np.sqrt(njk)+1)),labels=np.arange(0,int(np.sqrt(self.njk)),1))
        df_rand['jk_dec'] = pd.cut(df_rand['dec_rand'].values, np.linspace(self.cat.dec_gal.min(),self.cat.dec_gal.max(),int(np.sqrt(njk)+1)),labels=np.arange(0,int(np.sqrt(self.njk)),1))

        df_rand['jk_index'] = df_rand.jk_ra.values.astype(np.float) *int(np.sqrt(self.njk)) + df_rand.jk_dec.values.astype(np.float)        
        self.cat_rand = df_rand
        
        return df_rand
    
    def _calculate_correlation(self,dcat_ra1, dcat_ra2, dcat_dec1, dcat_dec2, rcat_ra, rcat_dec, jka, jkb, jkra):
        
        pairs = np.zeros(shape = (self.njk,self.njk,self.nangbins-1, 4))
        for x in self.bins_to_calculate:
            i,j = x

            ra_a_jk, dec_a_jk = dcat_ra1[jka==i], dcat_dec1[jka==i]
            ra_b_jk, dec_b_jk = dcat_ra2[jkb==j], dcat_dec2[jkb==j]

            ra_ra_a_jk, ra_dec_a_jk = rcat_ra[jkra==i], rcat_dec[jkra==i]
            ra_ra_b_jk, ra_dec_b_jk = rcat_ra[jkra==j], rcat_dec[jkra==j]

            ra_a_jk = ra_a_jk*np.cos(dec_a_jk*np.pi/180.)
            ra_b_jk = ra_b_jk*np.cos(dec_b_jk*np.pi/180.)
            ra_ra_a_jk = ra_ra_a_jk*np.cos(ra_dec_a_jk*np.pi/180.)
            ra_ra_b_jk = ra_ra_b_jk*np.cos(ra_dec_b_jk*np.pi/180.)

            tree_Da = cKDTree(np.array([ra_a_jk,dec_a_jk]).T)
            tree_Db = cKDTree(np.array([ra_b_jk,dec_b_jk]).T)
            tree_Ra = cKDTree(np.array([ra_a_jk, dec_ra_a_jk]).T)
            tree_Rb = cKDTree(np.array([ra_b_jk, dec_ra_b_jk]).T)

            DD = compute_tree_distances(tree_Da, tree_Db, max_sep, theta)
            DR = compute_tree_distances(tree_Da, tree_Rb, max_sep, theta)
            RD = compute_tree_distances(tree_Db, tree_Ra, max_sep, theta)
            RR = compute_tree_distances(tree_Ra, tree_Rb, max_sep, theta) 


            pairs_ = [DD, DR, RD, RR]
            pairs[i,j] = np.array(pairs_).T
            
        return pairs


    
    def _run_correlation(self,zrange1,zrange2, method = 'lz'):
        jackknifes = np.zeros(shape = (self.njk,self.nangbins-1))
        cat_a = self.cat[(self.cat.observed_redshift_gal>zrange1[0])&(self.cat.observed_redshift_gal<zrange1[1])]
        cat_b = self.cat[(self.cat.observed_redshift_gal>zrange2[0])&(self.cat.observed_redshift_gal<zrange2[1])]

        cat_rand = self._create_randcat(nrand=2*max(len(dfin_a),len(dfin_b)))

        #define catalogues 
        RA_a, dec_a, jk_a, Na_ = cat_a.ra_gal.values, cat_a.dec_gal.values, cat_a.jk_index.values, len(cat_a)
        RA_b, dec_b, jk_b, Nb_ = cat_b.ra_gal.values, cat_b.dec_gal.values, cat_b.jk_index.values, len(cat_b)
        RA_ra, dec_ra, jk_ra, NRA_ = cat_rand.ra_rand.values, cat_rand.dec_rand.values, cat_rand.jk_index.values, len(cat_rand)

        cat_lengths = [Na*Nb, Na*NRA, Nb*NRA, NRA*NRA]
        
        #objects in each jk region
        N_a,N_b, N_ra = self._NN(jk_a), self._NN(jk_b), self._NN(jk_ra)
        N = [np.multiply.outer(N_a, N_b), np.multiply.outer(N_a, N_ra),np.multiply.outer(N_b, N_ra), np.multiply.outer(N_ra, N_ra)]
        N = np.array(N)
        Na,Nb, NRA = Na_-np.array(N_a),Nb_-np.array(N_b), NRA_-np.array(N_ra)
        
        #pairs in each jk region
        N_1 = [Na*Nb, Na*NRA, NRA*Nb, NRA*(NRA-1)]
        N_1 = 0.5*np.array(N_1)
        
        
        pairs = self._calculate_correlation(RA_a, RA_b, dec_a, dec_b, RA_ra, dec_ra, jk_a, jk_b, jk_ra)
        
        
        for jk in range(self.njk):
            C = np.delete(pairs, jk, 0)
            C = np.delete(C, jk, 1)
            shp = C.shape
            C = C.reshape(shp[0]*shp[1], shp[2], shp[3])
            stack = np.sum(C, 0).T

            if method=='lz':
                DD, DR, RD, RR = np.array([stack[i]*N_1[0,jk]/N_1[i,jk] for i in range(len(N_1))])
                corr = (DD - DR - RD + RR)/RR 
                jackknifes[jk] = corr
                
            elif method=='simpler_estimator':

                N_jk_a = sum([N_a[i] for i in range(len(N_a)) if i != jk])
                N_jk_b = sum([N_b[i] for i in range(len(N_b)) if i != jk])
                Nra_jk= sum([N_ra[i] for i in range(len(N_ra)) if i != jk])

                DD, DR, RD, RR = np.array([stack[i] for i in range(len(N))])
                corr = (DD / RR) * (Nra_jk*(Nra_jk+1)) / (N_jk_a*(N_jk_b+1)) 
                jackknifes[jk] = corr
                
            return jackknifes
                

        
        
        
        
        