/*Options pour ecrire dans le log*/
OPTION NOTES SOURCE THREADS;

/*Déclaration des variables macro*/
%GLOBAL NBCPT NB_SC NB_AN_PROJECTION  NB_SC_INT NB_AN_PROJECTION_INT CHOC_CAPITAL HURDLE_RT;

%LET NBCPT = 100;
%LET NB_SC = 100;
%LET NB_AN_PROJECTION = 100;
%LET NB_SC_INT = 100;
%LET NB_AN_PROJECTION_INT = 100;
%LET CHOC_CAPITAL = 0.35;
%LET HURDLE_RT = 0.10;

*option et definition emplacement pour copier environnement dans les threads;
OPTIONS PRESENV SPOOL;

/******************************************/
/********** LIBRAIRIES UTILISÉES **********/
/******************************************/

*on prend toujours les tables en entree du repertoire de PROD;
%LET PATH = %str(\\ssq.local\Groupes\Actuariat_corporatif\6 - Projets\66 - D PDSA\021 - SAS Hackathon);

libname INPUT "&PATH.\Intrants Guy";
libname SERVEUR "&PATH.\Output";
libname MEMOIRE "&PATH.\Memlib" MEMLIB;

proc datasets library=MEMOIRE kill;run;


%MACRO DATA_STEP_CALCUL;

	************************************************************;
	*** Datastep pour le calcul des CF de toutes les boucles ***;
	************************************************************;

	*Datastep precompile;
	DATA MEMOIRE.CALCULS_3;

		********************************;
	    *** Création des hash tables ***;
	 	********************************;

		if _N_ = 1 then do;

			*Mortalité;
			declare hash h(dataset: "memoire.TX_DECES");
				h.defineKey('AGE');
				h.defineData('Qx');
				h.defineDone();
			call missing(Qx);

			*Lapses;
			declare hash g(dataset: "memoire.TX_RETRAIT");
				g.defineKey('an_proj');
				g.defineData('WX');
				g.defineDone();
			call missing(WX);

			*rendement;
			declare hash z(dataset: "MEMOIRE.RENDEMENT");
				z.defineKey('scn_proj','an_proj','TYPE');
				z.defineData('RENDEMENT');
				z.defineDone();
			call missing(RENDEMENT); 

			*Tx actualisation;
			declare hash a(dataset: "MEMOIRE.TX_INTERET");
				a.defineKey('an_proj');
				a.defineData('TX_ACTU');
				a.defineDone();
			call missing(TX_ACTU); 

			
			*Tx actualisation pour boucle interne;
			declare hash b(dataset: "MEMOIRE.TX_INTERET_INT");
				b.defineKey('an_eval');
				b.defineData('TX_ACTU_INT');
				b.defineDone();
			call missing(TX_ACTU_INT); 

		end;

		*****************************************************************************;
	    *** Retain sur variables importantes + initialisation des variables à t=0 ***;
		*****************************************************************************;

		retain 	MT_VM_PROJ MT_GAR_DECES_PROJ TX_SURVIE 0;

		set MEMOIRE.calculs_2;

		***********************************************;
		*** Initialisation des variables a lannee 0 ***;
		***********************************************;

		if an_eval = 0 and TYPE = "EXTERNE" then do;
			AGE						= age_deb;
			MT_VM_PROJ            	= MT_VM;
			MT_GAR_DECES_PROJ     	= MT_GAR_DECES;
			TX_SURVIE 				= 1;			
			TX_SURVIE_DEB 			= 1;				
			TX_ACTU 				= 1;
			QX 						= 0;
			WX 						= 0;
			an_proj 				= 0;

			COMMISSIONS= -TX_COMM_VENTE * MT_VM;
			VP_COMMISSIONS= COMMISSIONS;

			FRAIS_GEN = -FRAIS_ACQUI;
			VP_FRAIS_GEN= FRAIS_GEN;

			FLUX_NET= FRAIS_GEN + COMMISSIONS;
			VP_FLUX_NET= FLUX_NET;


			REVENUS= 0;	FRAIS_GEST= 0;	PMT_GARANTIE= 0;	
			VP_REVENUS= 0;	VP_FRAIS_GEST= 0; VP_PMT_GARANTIE= 0;

		end;

		else if an_eval_int = 0 and TYPE = "INTERNE" then do;

			if TYPE2 = "RESERVE" then	MT_VM_PROJ = MT_VM;
			else if TYPE2 = "CAPITAL" then	MT_VM_PROJ = MT_VM * (1 - &CHOC_CAPITAL.);

			AGE						= age_deb + an_eval;
			MT_GAR_DECES_PROJ     	= MT_GAR_DECES;
			TX_SURVIE 				= TX_SURVIE_DEB;			
			TX_ACTU 				= 1;
			QX 						= 0;
			WX 						= 0;
			an_proj 				= an_eval;

			COMMISSIONS= 0;
			VP_COMMISSIONS= 0;

			FRAIS_GEN = 0;
			VP_FRAIS_GEN= 0;

			FLUX_NET= 0;
			VP_FLUX_NET= 0;

			REVENUS= 0;	FRAIS_GEST= 0;	PMT_GARANTIE= 0;	
			VP_REVENUS= 0;	VP_FRAIS_GEST= 0; VP_PMT_GARANTIE= 0;

		end;

		*sil ny a plus de police ou bien si VM est a 0 et produit regulier on neffectu pas les calculs;
		else if TX_SURVIE = 0 or MT_VM_PROJ = 0 then delete;

		***********************************************************************;
	    *** Calcul des flux financiers pour toutes les années de projection ***;
		***********************************************************************;
		else do;

			scn_proj = ifn(TYPE = "INTERNE",scn_eval_int,scn_eval);
			*on incremente lage et lannee de projection;
			AGE = age_deb + an_eval + ifn(TYPE = "INTERNE",an_eval_int,0);
			an_proj = 0 + an_eval + ifn(TYPE = "INTERNE",an_eval_int,0);

			*projection de la valeur du fonds jusquà la fin de la periode evaluee, tous les flux sont en fin de periode;

			RENDEMENT=.;
			rc = z.find();

			MT_VM_DEB 	= MT_VM_PROJ;
			RENDEMENT 	= MT_VM_DEB * (RENDEMENT);
			FRAIS		= -(MT_VM_DEB + RENDEMENT / 2) * PC_REVENU_FDS;
			MT_VM_PROJ  = MT_VM_PROJ + RENDEMENT + FRAIS;

			*reinitialisaiton du montant garanti au deces;
			if FREQ_RESET_DECES = 1 and age <= MAX_RESET_DECES then MT_GAR_DECES_PROJ = MAX(MT_GAR_DECES_PROJ,MT_VM_PROJ);

			*taux de survie;
			Qx=.;
			rc = h.find();
			WX=.;
			rc = g.find();
			
			TX_SURVIE_DEB = TX_SURVIE;
			TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX);

			*flux;
			REVENUS 	 = - FRAIS * TX_SURVIE_DEB;
			FRAIS_GEST 	 = -(MT_VM_DEB + RENDEMENT / 2) * PC_HONORAIRES_GEST * TX_SURVIE_DEB;
			COMMISSIONS  = -(MT_VM_DEB + RENDEMENT / 2) * TX_COMM_MAINTIEN * TX_SURVIE_DEB;
			FRAIS_GEN 	 = - FRAIS_ADMIN * TX_SURVIE_DEB;
			PMT_GARANTIE = - MAX(0,MT_GAR_DECES_PROJ - MT_VM_PROJ) * QX * TX_SURVIE_DEB;
			FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE;

			*vp flux;
			TX_ACTU=.;
			rc = a.find();

			VP_REVENUS 	 	= REVENUS *  TX_ACTU;	
			VP_FRAIS_GEST 	= FRAIS_GEST *  TX_ACTU; 	
			VP_COMMISSIONS  = COMMISSIONS *  TX_ACTU; 
			VP_FRAIS_GEN 	= FRAIS_GEN *  TX_ACTU; 	
			VP_PMT_GARANTIE = PMT_GARANTIE *  TX_ACTU;
			VP_FLUX_NET 	= FLUX_NET *  TX_ACTU;

			*on ramene a la bonne periode pour la boucle interne;
			TX_ACTU_INT=.;
			rc = b.find();

			if TYPE = "INTERNE" and an_eval > 0 then do;
				VP_REVENUS 	 	= VP_REVENUS /  TX_ACTU_INT;	
				VP_FRAIS_GEST 	= VP_FRAIS_GEST /  TX_ACTU_INT; 	
				VP_COMMISSIONS  = VP_COMMISSIONS /  TX_ACTU_INT; 
				VP_FRAIS_GEN 	= VP_FRAIS_GEN /  TX_ACTU_INT; 	
				VP_PMT_GARANTIE = VP_PMT_GARANTIE /  TX_ACTU_INT;
				VP_FLUX_NET 	= VP_FLUX_NET /  TX_ACTU_INT;
			end;

		end;
	run;

%MEND DATA_STEP_CALCUL;


*****************************;
*** execution des calculs ***;
*****************************;

*Mise en mémoire de certaines tables;
DATA MEMOIRE.POPULATION;
	SET INPUT.POPULATION;
RUN;

DATA MEMOIRE.TX_DECES;
	SET INPUT.TX_DECES;
RUN;

DATA MEMOIRE.TX_RETRAIT;
	SET INPUT.TX_RETRAIT;
RUN;

DATA MEMOIRE.TX_INTERET;
	SET INPUT.TX_INTERET;
RUN;

DATA MEMOIRE.TX_INTERET_INT;
	SET INPUT.TX_INTERET_INT;
RUN;

DATA MEMOIRE.RENDEMENT;
	SET INPUT.RENDEMENT;
RUN;

DATA SERVEUR.CALCULS_SOMMAIRE(WHERE=(ID_COMPTE=0));
	ID_COMPTE = .; scn_eval=.; VP_FLUX_DISTRIBUABLES=.;
RUN;

*table bidon;
DATA MEMOIRE.SOMMAIRE_RESERVE(WHERE=(ID_COMPTE=0));
	ID_COMPTE=.; an_eval=.; scn_eval=.; VP_FLUX_NET=.;
RUN;

*on cesse dimprimer dans le log car il y a trop de lignes;
*option nonotes nosource;

%macro calculs;
	* on separe les calculs en plusieurs tables pour que ca soit plus performant (table entierement stockee en memoire);
	%do j = 1 %to &NBCPT.;

		* on explose la table de population en fonction du nb de scenario  et du nb dannee de projection pour faciliter le merge;
		data MEMOIRE.calculs_2;
			set MEMOIRE.POPULATION( where=(ID_COMPTE = &j.));

			TYPE = "EXTERNE";

			 do scn_eval = 1 to &NB_SC.;
			 	do an_eval = 0 to &NB_AN_PROJECTION.;

				*on conserve seulement les annees pertinentes;
		 		if age_deb + an_eval <= 99 then output;

			 	end;
			end;
		run;

		*Appel du datastep de calcul;
		%DATA_STEP_CALCUL;
		
		*CONSERVE LINFO;
		data MEMOIRE.FLUX_EXTERNE(keep = ID_COMPTE an_eval scn_eval REVENUS FRAIS_GEST COMMISSIONS FRAIS_GEN 
											 PMT_GARANTIE FLUX_NET) 
			MEMOIRE.POPULATION_INTERNE(drop=age RENDEMENT QX WX TX_ACTU);
			set MEMOIRE.calculs_3;
		run;

		proc datasets library=MEMOIRE;
	   		delete calculs_2 calculs_3;
		run;

		*manque de memoire alors on doit segmenter le calcul par type et simulation;
		%do m = 1 %to 2;

			%do k = 1 %to &NB_SC.;

					* on explose la table de population en fonction du nb de scenario  et du nb dannee de projection pour faciliter le merge;
					data MEMOIRE.calculs_2(drop=MT_VM_PROJ MT_GAR_DECES_PROJ TX_SURVIE);
						set MEMOIRE.POPULATION_INTERNE(where=(scn_eval = &k.));

						TYPE = "INTERNE";

						if &m. = 1 then TYPE2 = "RESERVE";
						else TYPE2 = "CAPITAL";

						do scn_eval_int = 1 to &NB_SC_INT.;

						 	do an_eval_int = 0 to &NB_AN_PROJECTION_INT.;

								*on reinitialyse les variables;
								MT_VM = MT_VM_PROJ;
								MT_GAR_DECES = MT_GAR_DECES_PROJ;
								TX_SURVIE_DEB = TX_SURVIE;

								*on conserve seulement les annees pertinentes;
						 		if age_deb + an_eval + an_eval_int <= 99 then output;

						 	end;
						end;
					run;

					*Appel du datastep de calcul;
					%DATA_STEP_CALCUL;
				
					*calcul reserve et capital;
					proc summary data = MEMOIRE.calculs_3 nway missing;
					 	class ID_COMPTE an_eval scn_eval scn_eval_int ; 
						var VP_FLUX_NET;
						output out = MEMOIRE.calculs_3(drop = _Type_ _freq_) sum=;
					run;
					
					proc summary data = MEMOIRE.calculs_3 nway missing;
					 	class ID_COMPTE an_eval scn_eval; 
						var VP_FLUX_NET;
						output out = MEMOIRE.calculs_3(drop = _Type_ _freq_) mean=;
					run;

					proc append base=MEMOIRE.SOMMAIRE_RESERVE DATA=MEMOIRE.calculs_3;RUN;

			%end;

		proc datasets library=MEMOIRE;
	   		delete calculs_3;
		run;

		*JUMELAGE AVEC BLOUCLE EXTERNE;
		data MEMOIRE.FLUX_EXTERNE(drop=VP_FLUX_NET rc);

			if _N_ = 1 then do;
				declare hash h(dataset: "memoire.SOMMAIRE_RESERVE");
				h.defineKey('ID_COMPTE','an_eval','scn_eval');
				h.defineData('VP_FLUX_NET');
				h.defineDone();
				call missing(VP_FLUX_NET);
			end;

			set MEMOIRE.FLUX_EXTERNE;

			VP_FLUX_NET=.;
			rc = h.find();
			%if &m. = 1 %then %do;
			RESERVE = VP_FLUX_NET;  
			%end;
			%if &m. = 2 %then %do;
			CAPITAL = VP_FLUX_NET -RESERVE;
			%end;

		run;

		proc datasets library=MEMOIRE;
	   		delete SOMMAIRE_RESERVE;
		run;

		%end;

		data MEMOIRE.FLUX_EXTERNE(drop=reserve_prec capital_prec );
			set MEMOIRE.FLUX_EXTERNE;

			reserve_prec = lag(reserve);
			capital_prec = lag(capital);

			if an_eval = 0 then do;
				PROFIT = FLUX_NET + RESERVE;
				FLUX_DISTRIBUABLES = (PROFIT + CAPITAL);
				VP_FLUX_DISTRIBUABLES = FLUX_DISTRIBUABLES / (1 + &HURDLE_RT.)**an_eval;
			end;
			else do;
				PROFIT = FLUX_NET + RESERVE - reserve_prec;
				FLUX_DISTRIBUABLES = (PROFIT + CAPITAL - capital_prec);
				VP_FLUX_DISTRIBUABLES = FLUX_DISTRIBUABLES / (1 + &HURDLE_RT.)**an_eval;
			end;
		run;
		
		proc summary data=MEMOIRE.FLUX_EXTERNE nway missing;
			class ID_COMPTE scn_eval;
			var VP_FLUX_DISTRIBUABLES;
			output out=MEMOIRE.FLUX_EXTERNE(drop = _type_ _freq_) sum=;
		run;
		
		*stockage des resultats;
		PROC APPEND BASE=SERVEUR.calculs_sommaire DATA=MEMOIRE.FLUX_EXTERNE(keep=ID_COMPTE scn_eval VP_FLUX_DISTRIBUABLES); 
		RUN;

		*les tables qui seront reutilisee sont supprimees;
		proc delete data  = MEMOIRE.calculs_2 MEMOIRE.calculs_3 MEMOIRE.reserve MEMOIRE.SOMMAIRE_RESERVE MEMOIRE.FLUX_EXTERNE; run;

	%end;

%mend;

%calculs;

*on remet limpression dans le log;
option notes source;

*On vide la mémoire;
LIBNAME MEMOIRE CLEAR;

