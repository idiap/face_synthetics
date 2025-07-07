#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import os
import enum

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Table, Float, Enum

import torch

from . import utils

# ---

databases = ('multipie', 'ffhq', 'webface42m')

# ---

class Database:
    """ Base class representing an sqlite based biometric image database. """
    Base = declarative_base()
    db_file = 'db.sqlite'
    database_name = 'db'

    # ---

    class Sample(Base):
        """ Samples table """
        __tablename__  = 'samples'
        id = Column(Integer, primary_key=True, autoincrement=True)
        key = Column(String(), nullable=False, unique=True)
        path = Column(String(), nullable=False, unique=False)
        identity = Column(Integer, nullable=False)
        meta_data = relationship('MetaData', backref='sample')
        groups = relationship('Group', secondary='groups_samples', back_populates='samples')

    # ---

    class MetaData(Base):
        """ Meta data table """
        __tablename__  = 'meta_data'
        id = Column(Integer, primary_key=True, autoincrement=True)
        value_str = Column(String())
        value_int = Column(Integer())
        value_float = Column(Float())
        sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
        type_id = Column(Integer, ForeignKey('meta_data_types.id'), nullable=False)

    # ---

    class MetaDataType(Base):
        """ Meta data types table """
        __tablename__  = 'meta_data_types'
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(), nullable=False, unique=True)
        class MetaDataTypeEnum(enum.Enum):
            str = 1
            int = 2
            float = 3
        type = Column(Enum(MetaDataTypeEnum))
        meta_data = relationship('MetaData', backref='type')

    # ---

    class Group(Base):
        __tablename__  = 'groups'
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(), nullable=False)
        protocol_id = Column(Integer, ForeignKey('protocols.id'), nullable=False)
        samples = relationship('Sample', secondary='groups_samples', back_populates='groups')

    # ---

    class Protocol(Base):
        __tablename__  = 'protocols'
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(), nullable=False, unique=True)
        groups = relationship('Group', backref='protocol')

    # ---

    groups_samples = Table(
        'groups_samples',
        Base.metadata,
        Column('group_id', ForeignKey('groups.id'), primary_key=True),
        Column('sample_id', ForeignKey('samples.id'), primary_key=True))
    
    # ---

    def __init__(
            self,
            read_only : bool = True,
            db_file_path : str = None
            ) -> None:
        self.read_only = read_only
        if db_file_path is None:
            self.db_file_path = utils.get_database_index_path(self.db_file)
        else:
            self.db_file_path = db_file_path
        self.engine = create_engine(f'sqlite:///{self.db_file_path}')
        self.db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        self.Base.query = self.db_session.query_property()
        self.Base.metadata.create_all(bind=self.engine)
        if read_only == True:
            self.data_directory = utils.get_database_data_directory(self.database_name)
        if self.read_only:
            def abort():
                raise RuntimeError('Database is read-only')
            self.db_session.flush = abort

    # ---

    def _add_protocol(
            self,
            protocol_name : str
            ) -> object:
        """ Add a protocol to the database, if not already present """
        assert self.read_only == False
        protocol = self.Protocol.query.filter_by(name=protocol_name).first()
        if protocol is None:
            protocol = self.Protocol(name=protocol_name)
            self.db_session.add(protocol)
            self.db_session.commit()
        return protocol
    
    # ---
    
    def _add_group(
            self,
            protocol : object,
            group_name : str,
            ) -> object:
        """ Add a group to a protocol, if not already present """
        assert self.read_only == False
        group = self.Group.query.filter_by(name=group_name, protocol_id=protocol.id).first()
        if group is None:
            group = self.Group(name=group_name, protocol_id=protocol.id)
            self.db_session.add(group)
            self.db_session.commit()
        return group

    # ---
    
    def _add_sample(
            self,
            group : object,
            key : str,
            path : str,
            identity : int,
            check_exist : bool = True,
            commit_session : bool = True,
            ) -> object:
        """ Add a sample to a group, if not already present """
        assert self.read_only == False
        if check_exist:
            sample = self.Sample.query.filter_by(key=key).first()
        else:
            sample = None
        if sample is None:
            sample = self.Sample(key=key, path=path, identity=identity)
            self.db_session.add(sample)
            if commit_session:
                self.db_session.commit()
        else:
            assert sample.identity == identity
        if sample not in group.samples:
            group.samples.append(sample)
            if commit_session:
                self.db_session.commit()
        return sample
    
    # ---
    
    def _add_meta_data_type(
            self,
            name : str,
            type : MetaDataType.MetaDataTypeEnum
            ) -> object:
        """ Add a metadata named type, if not already present """
        assert self.read_only == False
        meta_data_type = self.MetaDataType.query.filter_by(name=name).first()
        if meta_data_type is None:
            meta_data_type = self.MetaDataType(name=name, type=type)
            self.db_session.add(meta_data_type)
            self.db_session.commit()
        else:
            assert meta_data_type.type == type
        return meta_data_type
    
    # ---
    
    def _add_meta_data(
            self,
            sample : Sample,
            type : MetaDataType,
            value : str | int | float,
            check_exist : bool = True,
            commit_session : bool = True,
            ) -> object:
        """ Add a metadata to a sample, if not already present """
        assert self.read_only == False
        if check_exist:
            meta_data = self.MetaData.query.filter_by(sample_id=sample.id, type_id=type.id).first()
        else:
            meta_data = None
        if meta_data is None:
            if type.type == Database.MetaDataType.MetaDataTypeEnum.str:
                assert isinstance(value, str)
                meta_data = self.MetaData()
                meta_data.value_str = value
                sample.meta_data.append(meta_data)
                type.meta_data.append(meta_data)
            elif type.type == Database.MetaDataType.MetaDataTypeEnum.int:
                assert isinstance(value, int)
                meta_data = self.MetaData()
                meta_data.value_int = value
                sample.meta_data.append(meta_data)
                type.meta_data.append(meta_data)
            elif type.type == Database.MetaDataType.MetaDataTypeEnum.float:
                assert isinstance(value, float)
                meta_data = self.MetaData()
                meta_data.value_float = value
                sample.meta_data.append(meta_data)
                type.meta_data.append(meta_data)
            else:
                raise RuntimeError('Unknonwn metadata type')
            self.db_session.add(meta_data)
            if commit_session:
                self.db_session.commit()
        else:
            if type.type == Database.MetaDataType.MetaDataTypeEnum.str:
                assert isinstance(value, str)
                assert meta_data.value_str == value
            elif type.type == Database.MetaDataType.MetaDataTypeEnum.int:
                assert isinstance(value, int)
                assert meta_data.value_int == value
            elif type.type == Database.MetaDataType.MetaDataTypeEnum.float:
                assert isinstance(value, float)
                assert meta_data.value_float == value
            else:
                raise RuntimeError('Unknown metadata type')
        return meta_data
    
    # ---

    def list_protocols(
            self
            ) -> list[str]:
        """ List available protocols """
        protocols = self.Protocol.query.all()
        return [protocol.name for protocol in protocols]
    
    # ---
    
    def list_protocol_groups(
            self,
            protocol_name : str
            ) -> list[str]:
        """ List available groups for a given protocol """
        protocol = self.Protocol.query.filter_by(name=protocol_name).first()
        return [group.name for group in protocol.groups]
    
    # ---
    
    def query(
            self,
            protocol_names : list[str] | None = None,
            group_names : list[str] | None = None
            ) -> list[object]:
        """ Query samples given a list of protocols and a list of groups """
        if protocol_names is None:
            protocols = self.Protocol.query.all()
        else:
            assert isinstance(protocol_names, list)
            protocols = self.Protocol.query.filter(self.Protocol.name.in_(protocol_names)).all()
        protocol_ids = [protocol.id for protocol in protocols]
        if group_names is None:
            groups = self.Group.query.filter(self.Group.protocol_id.in_(protocol_ids)).all()
        else:
            groups = self.Group.query.filter(self.Group.protocol_id.in_(protocol_ids)).filter(self.Group.name.in_(group_names)).all()
        groups_ids = [group.id for group in groups]
        samples = self.Sample.query.filter(self.Sample.groups.any(self.Group.id.in_(groups_ids))).all()
        return samples
    
    # ---
    
    def load_sample(
            self,
            sample : Sample,
            device : torch.device = torch.device('cpu'),
            dtype : torch.dtype = torch.float32
            ) -> torch.Tensor:
        """ Return the data from a particular sample """
        file_path = os.path.join(self.data_directory, sample.path)
        sample_data = utils.load_image(
            file_path=file_path,
            device=device,
            dtype=dtype)
        return sample_data
    
    # ---
    
    def get_sample_metadata(
            self,
            sample : Sample
            ) -> dict:
        """ Return the metadata for a particular sample """
        meta_data : list[Database.MetaData] = self.MetaData.query.filter_by(sample_id=sample.id).all()
        metadata = {}
        for meta_datum in meta_data:
            if meta_datum.type.type == self.MetaDataType.MetaDataTypeEnum.str:
                metadata[meta_datum.type.name] = meta_datum.value_str
            elif meta_datum.type.type == self.MetaDataType.MetaDataTypeEnum.int:
                metadata[meta_datum.type.name] = int(meta_datum.value_int)
            elif meta_datum.type.type == self.MetaDataType.MetaDataTypeEnum.float:
                metadata[meta_datum.type.name] = float(meta_datum.value_float)
            else:
                raise RuntimeError('Unknonwn metadata type')
        return metadata

# ---

class MultipieDatabase(Database):
    """ MultiPIE database """
    db_file = 'multipie.sqlite'
    database_name = 'multipie'

    def __init__(self) -> None:
        super().__init__()

# ---

class FFHQDatabase(Database):
    """ FFHQ database """
    db_file = 'ffhq.sqlite'
    database_name = 'ffhq'

    def __init__(self) -> None:
        super().__init__()

# ---

class WebFace42MDatabase(Database):
    """ WebFace42M database """
    db_file = 'webface42m.sqlite'
    database_name = 'webface42m'

    def __init__(self) -> None:
        super().__init__()

class UTKfaceDatabase(Database):
    """ UTKface database """
    db_file = 'utkface_ageasis_train_subset_balanced_fullset_test.sqlite'
    database_name = 'utkface'

    def __init__(self) -> None:
        super().__init__()
